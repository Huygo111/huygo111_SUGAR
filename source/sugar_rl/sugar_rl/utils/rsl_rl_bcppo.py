import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks.mlp import MLP
import copy

class BCPPO(PPO):
    def __init__(self, policy,teacher_ckpt=None, **kwargs):
        super().__init__(policy, **kwargs)
        
        self.distill_loss_coef = 1.0
        self.bc_only_steps = 500
        self.critic_warmup_steps = 1000
        self.full_ppo_warmup_steps = 2000

        self.distill_loss_fn = nn.MSELoss()
        self.update_step = 0

        if teacher_ckpt is not None:
            print(f"[Distill] 正在从 {teacher_ckpt} 加载 Teacher 模型...")
            checkpoint = torch.load(teacher_ckpt, map_location=self.device)
            state_dict = checkpoint['model_state_dict']

            actor_weight_keys = sorted(
                [k for k in state_dict.keys() if k.startswith('actor.') and 'weight' in k],
                key=lambda x: int(x.split('.')[1])
            )

            num_obs = state_dict[actor_weight_keys[0]].shape[1]
            num_actions = state_dict[actor_weight_keys[-1]].shape[0]
            hidden_dims = [state_dict[k].shape[0] for k in actor_weight_keys[:-1]]

            print(f"[Distill] 检测到 Teacher 架构: 输入={num_obs}, 隐藏层={hidden_dims}, 输出={num_actions}")

            self.teacher_model = MLP(
                input_dim=num_obs,
                output_dim=num_actions,
                hidden_dims=hidden_dims,
                activation="elu",
            ).to(self.device)

            mlp_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('actor.'):
                    new_key = k.replace('actor.', '')
                    mlp_state_dict[new_key] = v
            
            self.teacher_model.load_state_dict(mlp_state_dict)
            self.teacher_model.eval()
            
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            self.teacher_std = checkpoint['model_state_dict']['std'].detach().to(self.device)
            print("[Distill] Teacher loaded successfully and parameters frozen.")
        else:
            self.teacher_model = None
            print("[Warning] No teacher_ckpt provided")
            assert False

    def update(self):  # noqa: C901
        if self.update_step >= self.bc_only_steps:
            self.schedule = "adaptive"
        else:
            self.schedule = "fixed"

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_distill_loss = 0 # Teacher 统计

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                assert False, "please check symmtry"
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_obs_list = []
                    for obs_group in self.policy.obs_groups["teacher"]:
                        teacher_obs_list.append(obs_batch[obs_group])
                        if torch.isnan(obs_batch[obs_group]).any():
                            assert False
                    teacher_obs_batch = torch.cat(teacher_obs_list, dim=-1)
                    # print("teacher obs batch shape:", teacher_obs_batch.shape)

                    teacher_action_mean = self.teacher_model(teacher_obs_batch)
                    teacher_action_std = self.teacher_std
                
                # # compute BC Loss (KL Divergence)
                # KL(T||S) = log(std_s/std_t) + (std_t^2 + (mu_t-mu_s)^2)/(2*std_s^2) - 0.5
                log_std_s = torch.log(sigma_batch + 1e-8)
                log_std_t = torch.log(teacher_action_std + 1e-8)
                distill_loss = (
                    log_std_s - log_std_t + 
                    (teacher_action_std.pow(2) + (teacher_action_mean - mu_batch).pow(2)) / (2.0 * (sigma_batch.pow(2)+1e-7)) - 
                    0.5
                ).sum(dim=-1).mean()
                mean_distill_loss += distill_loss.item()




            # =========================
            # Stage 1: Pure Distill
            # =========================
            if self.update_step < self.bc_only_steps:
                loss = self.distill_loss_coef * distill_loss

            # =========================
            # Stage 2: Distill + Critic
            # =========================
            elif self.update_step < self.critic_warmup_steps:
                alpha = min((self.update_step - self.bc_only_steps) / (self.critic_warmup_steps - self.bc_only_steps), 1.0)
                loss = (
                    self.distill_loss_coef * distill_loss 
                    + alpha * self.value_loss_coef * value_loss
                    )

            # =========================
            # Stage 3: Full PPO + Distill
            # =========================
            else:
                alpha = min((self.update_step - self.critic_warmup_steps) / (self.full_ppo_warmup_steps - self.critic_warmup_steps), 1.0)
                loss = (
                    surrogate_loss * alpha
                    + self.value_loss_coef * value_loss 
                    - self.entropy_coef * entropy_batch.mean() * alpha 
                    + self.distill_loss_coef * distill_loss * max(1.0-alpha, 0.0)
                )


            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_distill_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "distill": mean_distill_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.update_step += 1

        return loss_dict