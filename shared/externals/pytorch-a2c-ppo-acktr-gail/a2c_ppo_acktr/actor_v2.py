import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import FixedNormal
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.model import MLPBase, get_nonlinearity
from a2c_ppo_acktr.utils import AddBias, init
import json
import copy
import pdb

def get_joint_pos_vel(obs, base_state_len):
    # The first base_state_len dimensions are for the base state
    return obs[..., base_state_len:]

class IdentityDecoder(nn.Module):
    def __init__(self):
        super(IdentityDecoder, self).__init__()

    def forward(self, x, obs):
        return x

class LinearDecoder(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_):
        super(LinearDecoder, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, obs):
        return self.linear(x)


class IndexSelect(nn.Module):
    def __init__(self, indices, base_state_len=None):
        super(IndexSelect, self).__init__()
        self.torch_indices = torch.LongTensor(indices)
        self.base_state_len = base_state_len

    def forward(self, x, obs):
        if self.base_state_len is not None:
            x = get_joint_pos_vel(x, base_state_len=self.base_state_len)
        return x.index_select(-1, self.torch_indices.to(x.device))
    

def get_bilinear(x1, x2):
    outer_product = x1.unsqueeze(-2) * x2.unsqueeze(-1)
    return torch.cat([x1, x2, outer_product.flatten(start_dim=-2)], dim=-1)


class LinearDecoderWithObs(LinearDecoder):
    def __init__(self, num_inputs, num_outputs, init_, indices):
        super().__init__(num_inputs + len(indices), num_outputs, init_)

    def forward(self, x, partial_obs):
        input = torch.cat((partial_obs, x), dim=-1)
        return self.linear(input)


class MLPDecoder(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_hidden_layers, init_, 
    nonlinearity_mode="tanh", last_hidden_size=-1, last_nonlinearity=True):
        super(MLPDecoder, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        nonlinearity = get_nonlinearity(nonlinearity_mode)
        if last_hidden_size == -1:
            last_hidden_size = hidden_size 
        lst_modules = [init_(nn.Linear(num_inputs, hidden_size)), nonlinearity()]
        for _ in range(num_hidden_layers - 2):
            # Add more hidden layers in between the first and the last
            lst_modules.extend([init_(nn.Linear(hidden_size, hidden_size)), nonlinearity()])
        lst_modules.extend([init_(nn.Linear(hidden_size, last_hidden_size)), nonlinearity() if last_nonlinearity else nn.Identity()])
        self.model = nn.Sequential(*lst_modules)

        self.train()

    def forward(self, x, obs):
        return self.model(x)
    

class MLPDecoderWithObs(MLPDecoder):
    def __init__(self, num_inputs, hidden_size, num_hidden_layers, init_, 
                len_indices, nonlinearity_mode="tanh", last_hidden_size=-1, 
                last_nonlinearity=True, merge_mode="concat"):
        if merge_mode == "concat":
            input_size = num_inputs + len_indices
        elif merge_mode == "bilinear":
            input_size = num_inputs + len_indices + num_inputs * len_indices
        super(MLPDecoderWithObs, self).__init__(input_size, hidden_size, num_hidden_layers, init_,
         nonlinearity_mode=nonlinearity_mode, last_hidden_size=last_hidden_size, last_nonlinearity=last_nonlinearity)
        self.merge_mode = merge_mode


    def forward(self, x, partial_obs):
        if not hasattr(self, "merge_mode") or self.merge_mode == "concat":
            input = torch.cat((partial_obs, x), dim=-1)
        elif self.merge_mode == "bilinear":
            input = get_bilinear(partial_obs, x)
        return self.model(input)

class HierarchicalPolicyV2(nn.Module):
    def __init__(self, obs_shape, action_space, decoder_hidden_size=-1, module_hidden_size=32, decoder_num_hidden_layers=2,
        base=None, base_kwargs=None, base_state_len=16, share_within_mod=False, logstd_mode="separate", merge_mode="concat", unique_modules=False, 
        base_last_size_per_joint=-1, extend_local_state=False, hierarchy_json=None, full_state=True):
        super(HierarchicalPolicyV2, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        self.fix_low = False  # Whether or not the lower level policy is fixed or not
        nonlinearity_mode = "tanh" if "nonlinearity_mode" not in base_kwargs else base_kwargs["nonlinearity_mode"]
        self.base = base(obs_shape[0], **base_kwargs)
        self.base_logstd = AddBias(torch.zeros(self.base.last_hidden_size))  # Not used if self.fix_low is False
        num_inputs = self.base.output_size
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            init_ = lambda m: init(
                    m,
                    nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                    gain=0.01)
            if hierarchy_json is None:
                self.decoder = LinearDecoder(num_inputs, num_outputs, init_)
            else:
                self.decoder = LowPolicy(num_inputs, decoder_hidden_size, module_hidden_size, decoder_num_hidden_layers, num_outputs, init_, 
                            nonlinearity_mode=nonlinearity_mode, base_state_len=base_state_len, share_within_mod=share_within_mod, logstd_mode=logstd_mode, merge_mode=merge_mode, unique_modules=unique_modules, 
                            hierarchy_json=hierarchy_json, full_state=full_state, base_last_size_per_joint=base_last_size_per_joint, 
                            extend_local_state=extend_local_state)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
            if hierarchy_json is None:
                self.decoder = LinearDecoder(num_inputs, num_outputs, init_)
            else:
                self.decoder = LowPolicy(num_inputs, decoder_hidden_size, module_hidden_size, decoder_num_hidden_layers, num_outputs, init_, 
                            nonlinearity_mode=nonlinearity_mode, base_state_len=base_state_len, share_within_mod=share_within_mod, logstd_mode=logstd_mode, merge_mode=merge_mode, unique_modules=unique_modules, 
                            hierarchy_json=hierarchy_json, full_state=full_state, base_last_size_per_joint=base_last_size_per_joint, 
                            extend_local_state=extend_local_state)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
            if hierarchy_json is None:
                self.decoder = LinearDecoder(num_inputs, num_outputs, init_)
            else:
                self.decoder = LowPolicy(num_inputs, decoder_hidden_size, module_hidden_size, decoder_num_hidden_layers, num_outputs, init_, 
                            nonlinearity_mode=nonlinearity_mode, base_state_len=base_state_len, share_within_mod=share_within_mod,  logstd_mode=logstd_mode, merge_mode=merge_mode, unique_modules=unique_modules, 
                            hierarchy_json=hierarchy_json, full_state=full_state, base_last_size_per_joint=base_last_size_per_joint, 
                            extend_local_state=extend_local_state)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def get_controllers(self):
        return self.decoder.get_controllers()  

    def set_controllers(self, controllers, fix_weights=True, set_low_logstd=None, transfer_modules=None):
        self.decoder.set_controllers(controllers, 
                    fix_weights=fix_weights, set_low_logstd=set_low_logstd, transfer_modules=transfer_modules) 
    
    def set_noise(self, noise_args):
        self.decoder.set_noise(noise_args)  

    def get_noise_std(self):
        return self.decoder.get_noise_std()

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, return_feat=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.decoder(actor_features, inputs)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        if return_feat:
            return value, action, action_log_probs, actor_features, rnn_hxs
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action,  
                        return_feat=False, return_dist=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.decoder(actor_features, inputs)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        if return_feat:
            return value, action_log_probs, actor_features, dist_entropy, rnn_hxs
        if return_dist:
            return value, (action_log_probs, dist), actor_features, dist_entropy, rnn_hxs
        return value, action_log_probs, dist_entropy, rnn_hxs

    def evaluate_actions_sni(self, inputs, rnn_hxs, masks, action, noise_args=None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        self.decoder.set_noise(noise_args)
        dist_noise = self.decoder(actor_features, inputs)
        action_log_probs_noise = dist_noise.log_probs(action)
        dist_entropy_noise = dist_noise.entropy().mean()

        self.decoder.set_noise(None)
        dist = self.decoder(actor_features, inputs)
        self.decoder.set_noise(None)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        dist_entropy = (dist_entropy_noise + dist_entropy) / 2.
        return value, action_log_probs, action_log_probs_noise, dist_entropy, rnn_hxs

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative=False, is_train=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative = is_relative
        self.is_train = is_train
        self.noise = torch.tensor(0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            if self.is_relative:
                scale = self.sigma * x.detach() if not self.is_train else self.sigma * x
            else:
                scale = self.sigma
            sampled_noise = (self.noise.repeat(*x.size()).float().normal_()).to(x.device) * scale
            x = x + sampled_noise.to(x.device)
        return x 

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


def get_controllers(controller_types, decoder_hidden_size, module_hidden_size, decoder_num_hidden_layers, splice_output_size, init_,
                    nonlinearity_mode="tanh", full_state=False, extend_local_state=False, share_within_mod=False, logstd_mode="shared", 
                    base_last_size_per_joint=-1, merge_mode="concat"):
    controllers = {}
    for hierarchy_str, controller_type in controller_types.items():
        hierarchy = json.loads(hierarchy_str)
        if controller_type not in controllers:
            if not share_within_mod:
                modules = []
                for item in hierarchy:
                    if decoder_hidden_size == -1:
                        first_layer = IdentityDecoder()
                        module_input_size = splice_output_size
                    else:
                        module_input_size = decoder_hidden_size
                        first_layer = MLPDecoder(splice_output_size, decoder_hidden_size, decoder_num_hidden_layers, init_, nonlinearity_mode=nonlinearity_mode)
                    if base_last_size_per_joint != -1:
                        module_input_size = base_last_size_per_joint * len(hierarchy)
                    if isinstance(item, list):
                        controller_model = nn.ModuleList([
                            first_layer,
                            MLPDecoder(module_input_size, module_hidden_size, decoder_num_hidden_layers, init_, nonlinearity_mode=nonlinearity_mode)
                            ])
                        modules.append(controller_model)
                    else:
                        # If not full state, only get the angle and angular velocity.
                        # If full state, also get R and relative xyz coordinates(12 additional dimensions)
                        len_indices = 2 if not full_state else 14
                        if extend_local_state:
                            len_indices += 3 * (len(hierarchy) - 1)
                        controller_model = nn.ModuleList([
                            first_layer,
                            MLPDecoderWithObs(module_input_size, module_hidden_size, decoder_num_hidden_layers + 1, init_, len_indices,
                                                            nonlinearity_mode=nonlinearity_mode, last_hidden_size=1,  last_nonlinearity=False,
                                                            merge_mode=merge_mode)])
                        controller_std = None
                        if logstd_mode != "separate":
                            controller_std = AddBias(torch.zeros(1))
                        modules.append(nn.ModuleList([controller_model, controller_std]))
                controllers[controller_type] = nn.ModuleList(modules)
            else:
                # There is a single pair of MLP's that decodes the actions for all joints.
                len_indices = 2 * len(hierarchy) if not full_state else 14 * len(hierarchy)
                if decoder_hidden_size == -1:
                    first_layer = IdentityDecoder()
                    module_input_size = splice_output_size
                else:
                    first_layer = MLPDecoder(splice_output_size, decoder_hidden_size, decoder_num_hidden_layers, init_, nonlinearity_mode=nonlinearity_mode)
                    module_input_size = decoder_hidden_size
                if base_last_size_per_joint != -1:
                        module_input_size = base_last_size_per_joint * len(hierarchy)
                controller_model = nn.ModuleList([
                            first_layer,
                            MLPDecoderWithObs(module_input_size, module_hidden_size, decoder_num_hidden_layers + 1, init_, len_indices,
                                                            nonlinearity_mode=nonlinearity_mode, last_hidden_size=len(hierarchy), last_nonlinearity=False,
                                                            merge_mode=merge_mode)])
                controller_std = None
                if logstd_mode != "separate":
                    controller_std = AddBias(torch.zeros(len(hierarchy)))
                controllers[controller_type] =  nn.ModuleList([nn.ModuleList([controller_model, controller_std])])
    return controllers


class LowPolicy(nn.Module):
    def __init__(self, base_hidden_size, decoder_hidden_size, module_hidden_size, decoder_num_hidden_layers, num_outputs, init_, 
                base_state_len=16, base_last_size_per_joint=-1, nonlinearity_mode="tanh", share_within_mod=False, logstd_mode="shared", hierarchy_json=None, hierarchy_info=None, 
                merge_mode="concat", unique_modules=False, controllers=None, curr_controller=None,
                full_state=False, extend_local_state=False):
        super(LowPolicy, self).__init__()

        self.base_hidden_size = base_hidden_size
        self.base_last_size_per_joint = base_last_size_per_joint
        self.decoder_hidden_size = decoder_hidden_size
        self.module_hidden_size = module_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.base_state_len = base_state_len
        self.num_outputs = num_outputs  # Size of the action space 
        self.share_within_mod = share_within_mod
        self.top_level = False
        self.full_state = full_state  # Whether or not each joint's R matrix and relative position is available as well
        self.extend_local_state = extend_local_state
        self.nonlinearity_mode = nonlinearity_mode
        init_ = self.get_init()
        if hierarchy_json is not None:
            self.top_level = True
            with open(hierarchy_json, "r") as f:
                info = json.load(f)
            self.hierarchy = info["hierarchy"]
            self.controller_types = info["controller"]
            curr_input_size = base_hidden_size  # The inputs are the actor features 
        elif hierarchy_info is not None:
            self.hierarchy = hierarchy_info["hierarchy"]
            self.controller_types = hierarchy_info["controller"]
            curr_input_size = decoder_hidden_size  # The input is the signal from a module
        else:
            raise
        
        if unique_modules:
            # Modules are not reused 
            idx = 0
            for module in self.controller_types.keys():
                self.controller_types[module] = idx
                idx += 1
        
        self.input_size = curr_input_size
        self.splice_output_size = curr_input_size // len(self.hierarchy)
        if controllers is None:
            assert hierarchy_json is not None
            # Initialize the controllers at the top level of hierarchy
            self.controllers = get_controllers(self.controller_types, 
                        decoder_hidden_size, module_hidden_size, decoder_num_hidden_layers, self.splice_output_size, init_,
                        nonlinearity_mode=self.nonlinearity_mode, full_state=self.full_state, 
                        share_within_mod=self.share_within_mod, logstd_mode=logstd_mode, extend_local_state=self.extend_local_state, 
                        base_last_size_per_joint=self.base_last_size_per_joint, merge_mode=merge_mode)
        else:
            self.controllers = controllers

        # Option to not share logstd between same modules
        if logstd_mode == "separate" and self.top_level:
            self.logstd = AddBias(torch.zeros(num_outputs))
        self.add_noise = nn.Identity()
        self.noisy_min_clip = None
        self.noisy_max_clip = None

        # TODO: This probably needs to be changed with 3+ levels of hierarchy
        self.curr_controller = curr_controller
        if self.curr_controller is None:
            curr_controller = []
            indices_start = 0
            for _, module in enumerate(self.hierarchy):
                if base_last_size_per_joint != -1:
                    signal_size = base_last_size_per_joint * len(module)
                else:
                    signal_size = self.splice_output_size
                curr_controller.append(IndexSelect(np.arange(indices_start, indices_start+signal_size)))
                indices_start += signal_size
            self.curr_controller = nn.ModuleList(curr_controller)
        self.init_lst_modules()

    def get_init(self):
        return lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

    def init_lst_modules(self):
        lst_modules = []
        for idx, module in enumerate(self.hierarchy):
            if isinstance(module, list):
                if isinstance(module[0], list):
                    curr_controller = nn.ModuleList([IdentityDecoder() for _ in module])
                else:
                    module_str = str(module)
                    controller_type = self.controller_types[module_str]
                    curr_controller = self.controllers[controller_type]
                hierarchy_info = {"hierarchy": module, "controller": self.controller_types}
                lst_modules.append(LowPolicy(self.base_hidden_size, self.decoder_hidden_size, 
                                    self.module_hidden_size, self.decoder_num_hidden_layers, self.num_outputs, self.get_init(), 
                                    base_state_len=self.base_state_len, 
                                    hierarchy_info=hierarchy_info, controllers=self.controllers, 
                                    curr_controller=curr_controller, full_state=self.full_state,
                                    extend_local_state=self.extend_local_state,
                                    base_last_size_per_joint=self.base_last_size_per_joint,
                                    share_within_mod=self.share_within_mod))
            else:
                if not self.share_within_mod:
                    # This HierarchicalDecoder is a bottom-level decoder
                    # Select the position and the velocity
                    lst_indices = [module, module + self.num_outputs]
                    if self.full_state:
                        # The angles and angular velocities are both num_outputs
                        # length. The joint R and position is length 12
                        start_idx = 2 * self.num_outputs + 12 * module
                        end_idx = 2 * self.num_outputs + 12 * (module + 1)
                        lst_indices.extend([i for i in range(start_idx, end_idx)])
                    if self.extend_local_state:
                        for joint in self.hierarchy:
                            if joint != module:
                                # Get the rel coordinates of each joint. The coordinates
                                # follow the flattened global rotation matrix 
                                start_idx = 2 * self.num_outputs + 12 * joint + 9
                                end_idx = start_idx + 3
                                lst_indices.extend([i for i in range(start_idx, end_idx)])
                    lst_modules.append(IndexSelect(lst_indices, base_state_len=self.base_state_len))

        # Initialize modules for lower-level controllers. 
        if len(lst_modules) == 0 and self.share_within_mod:
            lst_indices = []
            for joint in self.hierarchy:
                lst_indices.extend([joint, joint + self.num_outputs])
                if self.full_state:
                    # The angles and angular velocities are both num_outputs
                    # length. The joint R and position is length 12
                    start_idx = 2 * self.num_outputs + 12 * joint
                    end_idx = 2 * self.num_outputs + 12 * (joint + 1)
                    lst_indices.extend([i for i in range(start_idx, end_idx)])
            lst_modules = [IndexSelect(lst_indices,  base_state_len=self.base_state_len)]
        self.lst_modules = nn.ModuleList(lst_modules)

    def get_controllers(self):
        return self.controllers    

    def set_controllers(self, controllers, fix_weights=True, 
                set_low_logstd=None, transfer_modules=None):
        if hasattr(self, "logstd") and set_low_logstd is not None:
            if transfer_modules is None:
                self.logstd._bias = nn.Parameter(torch.ones(self.logstd._bias.shape) * set_low_logstd)
            else:
                with torch.no_grad():
                    lst_types = [int(module) for module in transfer_modules.split(",")]
                    for joints, controller_type in self.controller_types.items():
                        lst_joints = json.loads(joints)
                        if controller_type in lst_types:
                            self.logstd._bias[lst_joints] = torch.ones(1) * set_low_logstd
        for key, module in controllers.items():
            # transfer_modules specifies which controllers will be transferred
            if transfer_modules is None or str(key) in transfer_modules.split(","):
                copy_module = copy.deepcopy(module)
                if fix_weights:
                    for param in copy_module.parameters():
                        param.requires_grad = False
                # Go through the module for each joint            
                for joint_module in copy_module:
                    if len(joint_module) == 2 and joint_module[1] is not None and isinstance(joint_module[1], AddBias):
                        if set_low_logstd is not None:
                            # Set low logstd manually
                            joint_module[1]._bias = nn.Parameter(torch.ones(joint_module[1]._bias.shape) * set_low_logstd)
                        joint_module[1]._bias.requires_grad = True
                assert len(copy_module) == len(self.controllers[key]), "Mismatch of transferred module size"
                self.controllers[key] = copy_module

        self.init_lst_modules()

    def set_noise(self, noise_args):
        if noise_args is None:
            self.add_noise = nn.Identity()
        else:
            self.add_noise = GaussianNoise(noise_args["noise_level"], 
                                        is_relative=noise_args["noise_relative"],
                                        is_train=noise_args["noise_train"])
            self.noisy_min_clip = noise_args["noisy_min_clip"]
            self.noisy_max_clip = noise_args["noisy_max_clip"]
    
    def get_noise_std(self):
        return self.add_noise.sigma

    def get_dist_from_dct(self,dist_dct):
        means = []
        logstds = []
        for joint in sorted(dist_dct.keys()):
            mean, logstd = dist_dct[joint]
            means.append(mean)
            if logstd is not None:
                logstds.append(logstd)
        if hasattr(self, "logstd"):
            means = torch.cat(means, dim=-1)
            zeros = torch.zeros(means.size())
            if means.is_cuda:
                zeros = zeros.cuda()
            action_logstd = self.logstd(zeros)
            dist = FixedNormal(means, action_logstd.exp())
        else:
            dist = FixedNormal(torch.cat(means, dim=-1), torch.cat(logstds, dim=-1).exp())
        return dist

    def forward(self, x, obs):
        output = {}
        new_input = self.add_noise(x)
        if self.noisy_min_clip is not None or self.noisy_max_clip is not None:
            new_input = torch.clamp(new_input, min=self.noisy_min_clip, max=self.noisy_max_clip)
        for controller, module, hierarchy_item in zip(self.curr_controller, self.lst_modules, self.hierarchy):
            if not isinstance(module, IndexSelect):
                # The controller indexes into the master controller's signal
                curr_output = controller(new_input, obs)
                output.update(module(curr_output, obs))
            else:
                controller_model, controller_log_std = controller
                # controller_model[0] is typically just an Identity(). module(obs, obs)
                # returns a chunk of the full embedding from the master controller
                action_mean = controller_model[1](controller_model[0](new_input, obs), module(obs, obs))
                # Get the logstd
                action_logstd = None
                # controller_log_std is only used if modules of the same type share the logstd vector
                if controller_log_std is not None:
                    zeros = torch.zeros(action_mean.size())
                    if action_mean.is_cuda:
                        zeros = zeros.cuda()
                    action_logstd = controller_log_std(zeros)
                if self.share_within_mod:
                    for item_idx, hierarchy_item in enumerate(self.hierarchy):
                        output[hierarchy_item] = (action_mean[..., item_idx].unsqueeze(-1), 
                                                  action_logstd[..., item_idx].unsqueeze(-1) if action_logstd is not None else None)
                else:
                    output[hierarchy_item] = (action_mean, action_logstd)
        if self.top_level:
            assert len(output) == self.num_outputs
            return self.get_dist_from_dct(output)
        return output