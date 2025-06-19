# From https://github.com/wjh720/QPLEX/, added here for convenience.
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.mamf import MAMF
from modules.mixers.momf import MOMF # Added import
import torch.nn.functional as F
import torch as th
from torch.optim import Adam
import numpy as np
from utils.rl_utils import build_td_lambda_targets
from utils.th_utils import get_parameters_num

class MAMF_MOMFLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "mamf_momf":
                self.mixer = MAMF(args)
                self.mixer_1 = MOMF(args, abs=False) # Changed from DMAQer(args, abs=False)
                self.mixer_2 = MOMF(args, abs=False) # Changed from DMAQer(args, abs=False)
                self.mixer_3 = MOMF(args, abs=False) # Changed from DMAQer(args, abs=False)
                self.mixer_4 = MOMF(args, abs=False) # Changed from DMAQer(args, abs=False)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.params += list(self.mixer_1.parameters())
            self.params += list(self.mixer_2.parameters())
            self.params += list(self.mixer_3.parameters())
            self.params += list(self.mixer_4.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            self.target_mixer_1 = copy.deepcopy(self.mixer_1)
            self.target_mixer_2 = copy.deepcopy(self.mixer_2)
            self.target_mixer_3 = copy.deepcopy(self.mixer_3)
            self.target_mixer_4 = copy.deepcopy(self.mixer_4)
            
        self.optimiser = Adam(params=self.params,  lr=args.lr)

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None):
        # Get the relevant quantities
        mixer_1 = self.mixer_1
        mixer_2 = self.mixer_2
        mixer_3 = self.mixer_3
        mixer_4 = self.mixer_4

        rewards = batch["reward"][:, :-1]
        rewards_ = rewards[:,:,0].unsqueeze(2)
        rewards_hurt = rewards[:,:,1].unsqueeze(2)
        rewards_kill = rewards[:,:,2].unsqueeze(2)
        rewards_survive = rewards[:,:,3].unsqueeze(2)
        rewards_win = rewards[:,:,4].unsqueeze(2)
        
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        mac_out = []
        mac_out_1 = []
        mac_out_2 = []
        mac_out_3 = []
        mac_out_4 = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_outs_1, agent_outs_2, agent_outs_3, agent_outs_4 = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_out_1.append(agent_outs_1)
            mac_out_2.append(agent_outs_2)
            mac_out_3.append(agent_outs_3)
            mac_out_4.append(agent_outs_4)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_out_1 = th.stack(mac_out_1, dim=1)
        mac_out_2 = th.stack(mac_out_2, dim=1)
        mac_out_3 = th.stack(mac_out_3, dim=1)
        mac_out_4 = th.stack(mac_out_4, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_3 = th.gather(mac_out_3[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_4 = th.gather(mac_out_4[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        
        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)
        
        x_mac_out_1 = mac_out_1.clone().detach()
        x_mac_out_1[avail_actions == 0] = -9999999
        max_action_qvals_1, max_action_index_1 = x_mac_out_1[:, :-1].max(dim=3)
        x_mac_out_2 = mac_out_2.clone().detach()
        x_mac_out_2[avail_actions == 0] = -9999999
        max_action_qvals_2, max_action_index_2 = x_mac_out_2[:, :-1].max(dim=3)
        x_mac_out_3 = mac_out_3.clone().detach()
        x_mac_out_3[avail_actions == 0] = -9999999
        max_action_qvals_3, max_action_index_3 = x_mac_out_3[:, :-1].max(dim=3)
        x_mac_out_4 = mac_out_4.clone().detach()
        x_mac_out_4[avail_actions == 0] = -9999999
        max_action_qvals_4, max_action_index_4 = x_mac_out_4[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()
            # self.logger.log_stat('agent_1_%d_q_1' % save_data[0], np.squeeze(q_data)[0], t_env)
            # self.logger.log_stat('agent_2_%d_q_2' % save_data[1], np.squeeze(q_data)[1], t_env)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_out_1 = []
        target_mac_out_2 = []
        target_mac_out_3 = []
        target_mac_out_4 = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_agent_outs_1, target_agent_outs_2, target_agent_outs_3, target_agent_outs_4 = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_out_1.append(target_agent_outs_1)
            target_mac_out_2.append(target_agent_outs_2)
            target_mac_out_3.append(target_agent_outs_3)
            target_mac_out_4.append(target_agent_outs_4)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        target_mac_out_1 = th.stack(target_mac_out_1, dim=1)
        target_mac_out_2 = th.stack(target_mac_out_2, dim=1)
        target_mac_out_3 = th.stack(target_mac_out_3, dim=1)
        target_mac_out_4 = th.stack(target_mac_out_4, dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_chosen_qvals_1 = th.gather(target_mac_out_1, 3, cur_max_actions).squeeze(3)
            target_chosen_qvals_2 = th.gather(target_mac_out_2, 3, cur_max_actions).squeeze(3)
            target_chosen_qvals_3 = th.gather(target_mac_out_3, 3, cur_max_actions).squeeze(3)
            target_chosen_qvals_4 = th.gather(target_mac_out_4, 3, cur_max_actions).squeeze(3)
            
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_max_qvals_1 = target_mac_out_1.max(dim=3)[0]
            target_max_qvals_2 = target_mac_out_2.max(dim=3)[0]
            target_max_qvals_3 = target_mac_out_3.max(dim=3)[0]
            target_max_qvals_4 = target_mac_out_4.max(dim=3)[0]

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            raise "Use Double Q"

        # Mix
        if mixer is not None:
            ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
            ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                            max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv
            
            # Adjusted for Mixer type
            chosen_action_qvals_hurt = mixer_1(chosen_action_qvals_1, batch["state"][:, :-1])
            chosen_action_qvals_kill = mixer_2(chosen_action_qvals_2, batch["state"][:, :-1])
            chosen_action_qvals_survive = mixer_3(chosen_action_qvals_3, batch["state"][:, :-1])
            chosen_action_qvals_win = mixer_4(chosen_action_qvals_4, batch["state"][:, :-1])

            if self.args.double_q:
                target_chosen = self.target_mixer(target_chosen_qvals, batch["state"], is_v=True)
                target_adv = self.target_mixer(target_chosen_qvals, batch["state"],
                                                actions=cur_max_actions_onehot,
                                                max_q_i=target_max_qvals, is_v=False)
                target_max_qvals = target_chosen + target_adv
                
                # Adjusted for Mixer type
                target_max_qvals_1 = self.target_mixer_1(target_chosen_qvals_1, batch["state"])
                target_max_qvals_2 = self.target_mixer_2(target_chosen_qvals_2, batch["state"])
                target_max_qvals_3 = self.target_mixer_3(target_chosen_qvals_3, batch["state"])
                target_max_qvals_4 = self.target_mixer_4(target_chosen_qvals_4, batch["state"])
            else:
                raise "Use Double Q"

        # Calculate 1-step Q-Learning targets
        targets = build_td_lambda_targets(rewards_, terminated, mask, target_max_qvals, 
                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
        targets_hurt = build_td_lambda_targets(rewards_hurt, terminated, mask, target_max_qvals_1, 
                            self.args.n_agents, self.args.gamma, self.args.td_lambda)
        targets_kill = build_td_lambda_targets(rewards_kill, terminated, mask, target_max_qvals_2,
                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
        targets_survive = build_td_lambda_targets(rewards_survive, terminated, mask, target_max_qvals_3, 
                    self.args.n_agents, self.args.gamma, self.args.td_lambda)
        targets_win = build_td_lambda_targets(rewards_win, terminated, mask, target_max_qvals_4,
                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)
        
        td_error_hurt = (chosen_action_qvals_hurt - targets_hurt.detach())
        td_error_hurt = 0.5 * td_error_hurt.pow(2)

        td_error_kill = (chosen_action_qvals_kill - targets_kill.detach())
        td_error_kill = 0.5 * td_error_kill.pow(2)
        
        td_error_survive = (chosen_action_qvals_survive - targets_survive.detach())
        td_error_survive = 0.5 * td_error_survive.pow(2)

        td_error_win = (chosen_action_qvals_win - targets_win.detach())
        td_error_win = 0.5 * td_error_win.pow(2)

        mask1 = mask.expand_as(td_error)
        masked_td_error = td_error * mask1
        
        mask2 = mask.expand_as(td_error_hurt)
        masked_td_error_hurt = td_error_hurt * mask2

        mask3 = mask.expand_as(td_error_kill)
        masked_td_error_kill = td_error_kill * mask3
        
        mask4 = mask.expand_as(td_error_survive)
        masked_td_error_survive = td_error_survive * mask4

        mask5 = mask.expand_as(td_error_win)
        masked_td_error_win = td_error_win * mask5
        
        # mask2 = mask.expand_as(mo_loss)
        # masked_mo_loss = mo_loss * mask2
        
        # l_obj_dis = masked_mo_loss.sum() / mask.sum()
        L_td = masked_td_error.sum() / mask.sum()
        L_td_hurt = masked_td_error_hurt.sum() / mask.sum()
        L_td_kill = masked_td_error_kill.sum() / mask.sum()
        L_td_survive = masked_td_error_survive.sum() / mask.sum()
        L_td_win = masked_td_error_win.sum() / mask.sum()
        

        l_obj_dis_w = 0.6 ** (t_env / 200000)
        l_s_h = L_td_hurt + L_td_kill + L_td_survive + 0.1*L_td_win
        L_mo = l_s_h #*l_obj_dis_w
        # L_td = (1-l_obj_dis_w)*L_td
        loss = L_td + 0.2*L_mo

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                       show_demo=show_demo, save_data=save_data)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.mixer_1 is not None and self.target_mixer_1 is not None: # Added check for target_mixer_1
            self.target_mixer_1.load_state_dict(self.mixer_1.state_dict())
        if self.mixer_2 is not None and self.target_mixer_2 is not None: # Added check for target_mixer_2
            self.target_mixer_2.load_state_dict(self.mixer_2.state_dict())
        if self.mixer_3 is not None and self.target_mixer_3 is not None: # Added check for target_mixer_3
            self.target_mixer_3.load_state_dict(self.mixer_3.state_dict())
        if self.mixer_4 is not None and self.target_mixer_4 is not None: # Added check for target_mixer_4
            self.target_mixer_4.load_state_dict(self.mixer_4.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.mixer_1 is not None: # Added check
            self.mixer_1.cuda()
        if self.target_mixer_1 is not None: # Added check
            self.target_mixer_1.cuda()
        if self.mixer_2 is not None: # Added check
            self.mixer_2.cuda()
        if self.target_mixer_2 is not None: # Added check
            self.target_mixer_2.cuda()
        if self.mixer_3 is not None: # Added check
            self.mixer_3.cuda()
        if self.target_mixer_3 is not None: # Added check
            self.target_mixer_3.cuda()
        if self.mixer_4 is not None: # Added check
            self.mixer_4.cuda()
        if self.target_mixer_4 is not None: # Added check
            self.target_mixer_4.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
