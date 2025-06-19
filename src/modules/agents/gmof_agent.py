
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init

class GMOFAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GMOFAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape + args.n_agents * args.n_actions, args.rnn_hidden_dim * args.n_actions * 2),
                                        nn.ELU(inplace=True),
                                        nn.Linear(args.rnn_hidden_dim * args.n_actions * 2, args.rnn_hidden_dim * args.n_actions))

        self.hyper_mo_balance = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                                        nn.ELU(inplace=True),
                                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                                        nn.Tanh()
                                        )

        self.hyper_w1 = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))  
        self.hyper_b1 = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))

        self.hyper_w2 = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))  
        self.hyper_b2 = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))

        self.hyper_w3 = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))  
        self.hyper_b3 = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))

        self.hyper_w4 = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))  
        self.hyper_b4 = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))

        # self.hyper_w5 = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))  
        # self.hyper_b5 = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))

        self.hyper_mo_f = nn.Sequential(nn.Linear(args.hypernet_embed, 4), nn.Tanh())
        self.hyper_mo_b = nn.Sequential(nn.Linear(args.hypernet_embed, 1), nn.Tanh())
        
        # 应用Xavier初始化
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.GRUCell):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)
                elif 'bias' in name:
                    init.zeros_(param)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        
        x = F.elu(self.fc1(inputs.view(-1, e)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        h_detach = h.detach()

        mo_balance = self.hyper_mo_balance(h_detach)
        mo_f = self.hyper_mo_f(mo_balance).reshape(-1, 4, 1)
        mo_b = self.hyper_mo_b(mo_balance).reshape(-1, 1, 1)
        mo_b_norm = mo_b
        mo_f_norm = mo_f

        mo_balance_detach = mo_balance.detach()
        w1 = self.hyper_w1(mo_balance_detach).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        w2 = self.hyper_w2(mo_balance_detach).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        w3 = self.hyper_w3(mo_balance_detach).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        w4 = self.hyper_w4(mo_balance_detach).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        # w5 = self.hyper_w5(mo_balance_detach).reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        b1 = self.hyper_b1(mo_balance_detach).reshape(-1, 1, self.args.n_actions)
        b2 = self.hyper_b2(mo_balance_detach).reshape(-1, 1, self.args.n_actions)
        b3 = self.hyper_b3(mo_balance_detach).reshape(-1, 1, self.args.n_actions)
        b4 = self.hyper_b4(mo_balance_detach).reshape(-1, 1, self.args.n_actions)
        # b5 = self.hyper_b5(mo_balance_detach).reshape(-1, 1, self.args.n_actions)
        
        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        q1 = th.matmul(h, w1) + b1
        q2 = th.matmul(h, w2) + b2
        q3 = th.matmul(h, w3) + b3
        q4 = th.matmul(h, w4) + b4
        # q5 = th.matmul(h, w5) + b5
        # q1_detach = q1.detach()
        # q2_detach = q2.detach()
        q1_detach = q1
        q2_detach = q2
        q3_detach = q3
        q4_detach = q4
        # q5_detach = q5
        q_ = th.cat([q1_detach, q2_detach, q3_detach, q4_detach], dim=1).reshape(b*a, self.args.n_actions, 4)
        # q_ = th.cat([q1_detach, q2_detach, q3], dim=1).reshape(b*a, self.args.n_actions, 3)
        q = th.matmul(q_, mo_f_norm) + mo_b_norm # mo_b


        return q.view(b, a, -1), h.view(b, a, -1), q1.view(b, a, -1), q2.view(b, a, -1), q3.view(b, a, -1), q4.view(b, a, -1), mo_f.view(b, a, -1)