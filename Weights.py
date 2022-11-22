import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

#File Imports - start
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SharedDot(nn.Module):
    def __init__(self, in_features, out_features, n_channels, bias=False,
                 init_weight=None, init_bias=None):
        super(SharedDot, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.weight = nn.Parameter(torch.Tensor(n_channels, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weight:
            nn.init.uniform_(self.weight.data, a=-self.init_weight, b=self.init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight.data, a=0.)
        if self.bias is not None:
            if self.init_bias:
                nn.init.constant_(self.bias.data, self.init_bias)
            else:
                nn.init.constant_(self.bias.data, 0.)

    def forward(self, input):
        output = torch.matmul(self.weight, input.unsqueeze(1))
        if self.bias is not None:
            output.add_(self.bias.unsqueeze(0).unsqueeze(3))
        output.squeeze_(1)
        return output

class FeatureEncoder(nn.Module):
    def __init__(self, n_layers, in_features, latent_space_size,
                 deterministic=False, batch_norm=True,
                 mu_weight_std=0.001, mu_bias=0.0,
                 logvar_weight_std=0.01, logvar_bias=0.0,
                 easy_init=False):
        super(FeatureEncoder, self).__init__()
        self.n_layers = n_layers
        self.in_features = in_features
        self.latent_space_size = latent_space_size
        self.deterministic = deterministic
        self.batch_norm = batch_norm
        self.mu_weight_std = mu_weight_std
        self.mu_bias = mu_bias
        self.logvar_weight_std = logvar_weight_std
        self.logvar_bias = logvar_bias
        self.easy_init = easy_init

        if n_layers > 0:
            self.features = nn.Sequential()
            for i in range(n_layers):
                self.features.add_module('mlp{}'.format(i), nn.Linear(in_features, in_features, bias=False))
                if self.batch_norm:
                    self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(in_features))
                self.features.add_module('mlp{}_swish'.format(i), Swish())

        self.mus = nn.Sequential(OrderedDict([
            ('mu_mlp0', nn.Linear(in_features, latent_space_size, bias=True))
        ]))
        if not easy_init:
            with torch.no_grad():
                self.mus[-1].weight.data.normal_(std=mu_weight_std)
                nn.init.constant_(self.mus[-1].bias.data, self.mu_bias)

        if not self.deterministic:
            self.logvars = nn.Sequential(OrderedDict([
                ('logvar_mlp0', nn.Linear(in_features, latent_space_size, bias=True))
            ]))
            if not easy_init:
                with torch.no_grad():
                    self.logvars[-1].weight.data.normal_(std=logvar_weight_std)
                    nn.init.constant_(self.logvars[-1].bias.data, self.logvar_bias)

    def forward(self, input):
        if self.n_layers > 0:
            features = self.features(input)
        else:
            features = input

        if self.deterministic:
            return self.mus(features)
        else:
            return self.mus(features), self.logvars(features)

class WeightsEncoder(FeatureEncoder):
    def forward(self, input):
        mus = super().forward(input)
        weights = nn.functional.log_softmax(mus, dim=1)
        print("Weights from WeightEncoder is", weights)
        return weights

class LocalCondRNVPDecoder(nn.Module):
    def __init__(self, n_flows, f_n_features, g_n_features, weight_std=0.01):
        super(LocalCondRNVPDecoder, self).__init__()
        self.n_flows = n_flows
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std

        self.flows = nn.ModuleList(
            [CondRealNVPFlow3DTriple(f_n_features, g_n_features,
                                     weight_std=self.weight_std, pattern=(i % 2)) for i in range(n_flows)]
        )

    @staticmethod
    def get_param_count(n_flows, f_n_features, g_n_features):
        count_CondRealNVPFlow3D = 18 * f_n_features + 4 * f_n_features * g_n_features + 6 * f_n_features**2
        count_CondRealNVPFlow3DTriple = 3 * count_CondRealNVPFlow3D
        total_count = n_flows * count_CondRealNVPFlow3DTriple
        return total_count

    def forward(self, p, g, mode='direct'):
        ps = []
        mus = []
        logvars = []
        for i in range(self.n_flows):
            if mode == 'direct':
                cur_p = p if i == 0 else ps[-1]
                buf = self.flows[i](cur_p, g, mode=mode)
                ps = ps + buf[0]
                mus = mus + buf[1]
                logvars = logvars + buf[2]
            elif mode == 'inverse':
                cur_p = p if i == 0 else ps[0]
                buf = self.flows[-(i + 1)](cur_p, g, mode=mode)
                ps = buf[0] + ps
                mus = buf[1] + mus
                logvars = buf[2] + logvars

        return ps, mus, logvars

class CondRealNVPFlow3DTriple(nn.Module):
    def __init__(self, f_n_features, g_n_features, weight_std=0.02, pattern=0, centered_translation=False):
        super(CondRealNVPFlow3DTriple, self).__init__()
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.pattern = pattern
        self.centered_translation = centered_translation

        if pattern == 0:
            self.nvp1 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0],
                                          centered_translation=centered_translation)
            self.nvp2 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[1],
                                          centered_translation=centered_translation)
            self.nvp3 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[2],
                                          centered_translation=centered_translation)
        elif pattern == 1:
            self.nvp1 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0, 1],
                                          centered_translation=centered_translation)
            self.nvp2 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[0, 2],
                                          centered_translation=centered_translation)
            self.nvp3 = CondRealNVPFlow3D(f_n_features, g_n_features,
                                          weight_std=weight_std, warp_inds=[1, 2],
                                          centered_translation=centered_translation)

    def forward(self, p, g, mode='direct'):
        if mode == 'direct':
            p1, mu1, logvar1 = self.nvp1(p, g, mode=mode)
            p2, mu2, logvar2 = self.nvp2(p1, g, mode=mode)
            p3, mu3, logvar3 = self.nvp3(p2, g, mode=mode)
        elif mode == 'inverse':
            p3, mu3, logvar3 = self.nvp3(p, g, mode=mode)
            p2, mu2, logvar2 = self.nvp2(p3, g, mode=mode)
            p1, mu1, logvar1 = self.nvp1(p2, g, mode=mode)

        return [p1, p2, p3], [mu1, mu2, mu3], [logvar1, logvar2, logvar3]

class CondRealNVPFlow3D(nn.Module):
    def __init__(self, f_n_features, g_n_features,
                 weight_std=0.01, warp_inds=[0],
                 centered_translation=False, eps=1e-6):
        super(CondRealNVPFlow3D, self).__init__()
        self.f_n_features = f_n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = [0, 1, 2]
        self.centered_translation = centered_translation
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        self.T_mu_0 = nn.Sequential(OrderedDict([
            ('mu_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1)),
            ('mu_sd0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd0_relu', nn.ReLU(inplace=True)),
            ('mu_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)),
            ('mu_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False))
        ]))

        self.T_mu_0_cond_w = nn.Sequential(OrderedDict([
            ('mu_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('mu_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd1_film_w0_swish', Swish()),
            ('mu_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_mu_0_cond_b = nn.Sequential(OrderedDict([
            ('mu_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('mu_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('mu_sd1_film_b0_swish', Swish()),
            ('mu_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_mu_1 = nn.Sequential(OrderedDict([
            ('mu_sd1_relu', nn.ReLU(inplace=True)),
            ('mu_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_mu_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_w[-1].bias.data, 0.0)
            self.T_mu_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_b[-1].bias.data, 0.0)
            self.T_mu_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_1[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_sd0', SharedDot(len(self.keep_inds), self.f_n_features, 1)),
            ('logvar_sd0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd0_relu', nn.ReLU(inplace=True)),
            ('logvar_sd1', SharedDot(self.f_n_features, self.f_n_features, 1)),
            ('logvar_sd1_bn', nn.BatchNorm1d(self.f_n_features, affine=False))
        ]))

        self.T_logvar_0_cond_w = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_w0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('logvar_sd1_film_w0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd1_film_w0_swish', Swish()),
            ('logvar_sd1_film_w1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_logvar_0_cond_b = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_b0', nn.Linear(self.g_n_features, self.f_n_features, bias=False)),
            ('logvar_sd1_film_b0_bn', nn.BatchNorm1d(self.f_n_features)),
            ('logvar_sd1_film_b0_swish', Swish()),
            ('logvar_sd1_film_b1', nn.Linear(self.f_n_features, self.f_n_features, bias=True))
        ]))

        self.T_logvar_1 = nn.Sequential(OrderedDict([
            ('logvar_sd1_relu', nn.ReLU(inplace=True)),
            ('logvar_sd2', SharedDot(self.f_n_features, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_logvar_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_w[-1].bias.data, 0.0)
            self.T_logvar_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_b[-1].bias.data, 0.0)
            self.T_logvar_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_1[-1].bias.data, 0.0)

    def forward(self, p, g, mode='direct'):
        logvar = torch.zeros_like(p)
        mu = torch.zeros_like(p)

        logvar[:, self.warp_inds, :] = nn.functional.softsign(self.T_logvar_1(
            torch.add(self.eps, torch.exp(self.T_logvar_0_cond_w(g).unsqueeze(2))) *
            self.T_logvar_0(p[:, self.keep_inds, :].contiguous()) + self.T_logvar_0_cond_b(g).unsqueeze(2)
        ))

        mu[:, self.warp_inds, :] = self.T_mu_1(
            torch.add(self.eps, torch.exp(self.T_mu_0_cond_w(g).unsqueeze(2))) *
            self.T_mu_0(p[:, self.keep_inds, :].contiguous()) + self.T_mu_0_cond_b(g).unsqueeze(2)
        )

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            p_out = torch.sqrt(torch.add(self.eps, torch.exp(logvar))) * p + mu
        elif mode == 'inverse':
            p_out = (p - mu) / torch.sqrt(torch.add(self.eps, torch.exp(logvar)))

        return p_out, mu, logvar

#File imports - end

#Generation and usage of weights

#Variables
g_latent_space_size = 128
n_components = 4
p_latent_space_size = 3
p_prior_n_layers = 1
p_decoder_n_features = 64
p_decoder_n_flows = 21

#Networks as objects
p_prior =  FeatureEncoder(p_prior_n_layers, g_latent_space_size,
                                          p_latent_space_size, deterministic=False,
                                          mu_weight_std=0.001, mu_bias=0.0,
                                          logvar_weight_std=0.01, logvar_bias=0.0)

mixture_weights_encoder = WeightsEncoder(3, g_latent_space_size,
                                          n_components, deterministic=True,
                                          mu_weight_std=0.001, mu_bias=0.0,
                                          logvar_weight_std=0.01, logvar_bias=0.0)
pc_decoder = LocalCondRNVPDecoder(p_decoder_n_flows,
                                               p_decoder_n_features,
                                               g_latent_space_size,
                                               weight_std=0.01)


#Functions from Flow Model Mixture
def one_flow_decode(p_input, g_sample, pc_decoder, n_sampled_points):
        '''
        decode flow for one flow only
        Args:
            p_input: input point cloud
            g_sample: another input point cloud resampled from the same point cloud like p_input
            pc_decoder: decoder flow
            n_sampled_points: number of points need to be sampled
        Returns:
            output: p_prior_samples: the output decoder flow samples list
                    p_prior_mus: the output decoder flow mus list
                    p_prior_logvars: the output decoder flow log deviations list
        '''
        output = {}
        # for training/generation task
        output['p_prior_mus'], output['p_prior_logvars'] = p_prior(g_sample)
        output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
                g_sample.shape[0], p_latent_space_size, n_sampled_points
            )]
        output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
                g_sample.shape[0], p_latent_space_size, n_sampled_points
            )]
        
        #train decoder flow
        buf = pc_decoder(p_input, g_sample, mode='inverse')
        output['p_prior_samples'] = buf[0] + [p_input]

        output['p_prior_mus'] += buf[1]
        output['p_prior_logvars'] += buf[2]

        return output

def get_weights(g_sample):
        '''
        decide the weights of all flows
        Args:
            g_sample: input point cloud
            warmup: if use warmup, then in the first few epochs, we use global weights type
                    else, we use learned weights type.
        Returns:
            mixture_weights_logits: log weights of each flow in decoder flows.
        '''
        mixture_weights_logits = mixture_weights_encoder(g_sample)
        return mixture_weights_logits

def decode(p_input, g_sample, n_sampled_points, labeled_samples=False, warmup=False):
        '''
        mixtures of flows in decoder.

        Args:
            p_input: input point cloud  B * 3 * N
            g_sample: another sampled point cloud, from the same shape as p_input   B * 3 * N
            n_sampled_points: number of sampled points, when training,it's the number of points in p_input.
                              when evaluation, it's 2048 for generation /autoencoding, 2500 for svr.
            labeled_samples: if true, output labels (each point belongs to which flow), used in evaluation
                             if false, only output generated point cloud, and the mixtures weights
            warmup: if true, use global weights at first
                    else, use learned weights
        Returns:
            samples: output point clouds with labels
            labels: point labels
            mixture_weights_logits: weight of each flow
            output_decoder: output point clouds list
        '''
        mixture_weights_logits = get_weights(g_sample)
        sampled_cloud_size = [n_sampled_points for _ in range(n_components)]
        output_decoder = []
        for i in range(n_components):
            #generate output parts for each flow decoder
            one_decoder = one_flow_decode(p_input, g_sample, pc_decoder[i], sampled_cloud_size[i])
            output_decoder.append(one_decoder)  
        return output_decoder, mixture_weights_logits


p_input = torch.rand(5, 3, 2048)
n_sampled_points = 2048
g_sample = torch.rand(5, 3, 2048)

get_weights(p_input)
