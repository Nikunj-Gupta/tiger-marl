import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv as GAT
from components.tiger_tgat import TGANMARL
from components.tiger_graph import NeighborFinder
import random


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = int(np.prod(args.obs_shape))

        self.hidden_dim = self.obs_dim

        # GAT for initial edge attention scoring
        self.gat = GAT(self.hidden_dim, self.hidden_dim, heads=1, concat=False)

        # TGAT: temporal graph attention network
        ngh_finder = NeighborFinder(adj_list=[[] for _ in range(self.n_agents + 1)])
        self.tgan = TGANMARL(ngh_finder, self.hidden_dim)

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        combined_dim = self.state_dim + self.n_agents * self.hidden_dim

        self.V = nn.Sequential(
            nn.Linear(combined_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def generate_edges_with_reset_timesteps_no_interlinks(self, N, g, k, t, neighbor_table):
        """
        Generate temporal edges for fully connected subgraphs.

        Within each episode, builds edges between agents at the current timestep
        and their neighbors in past timesteps (controlled by k_past_neighbors and
        self_past hyperparameters), resetting after every t timesteps.

        Args:
            N: Total number of nodes (batch_size * timesteps * n_agents)
            g: Number of agents per timestep
            k: Number of past neighbor layers (unused, kept for API compatibility)
            t: Number of timesteps per episode (reset interval)
            neighbor_table: Dict mapping node index -> list of neighbor indices

        Returns:
            sorted_edges: List of (src, dst) edge tuples
            timestep_values: Corresponding timestep for each edge
        """
        edges = set()
        timesteps = {}

        k_past_self = self.args.self_past
        k_past_neighbors = self.args.k_past_neighbors

        # Cross-agent temporal edges: connect agents to their neighbors in past timesteps
        for batch in range(int((N / g) / t)):
            start_node = batch * (g * t)
            for reverse_timestep in range(t):
                timestep = t - reverse_timestep - 1
                curr_start = start_node + timestep * g
                curr_end = curr_start + g
                for node in range(curr_start, curr_end):
                    for j in range(max(0, timestep - k_past_neighbors), timestep + 1):
                        for neighbor in neighbor_table[node]:
                            past_neighbor = neighbor - (timestep - j) * g
                            past_start = start_node + j * g
                            past_end = past_start + g
                            if past_start <= past_neighbor < past_end:
                                edge = (node, past_neighbor)
                                edges.add(edge)
                                timesteps[edge] = int(node / g)

        # Self-temporal edges: connect each agent to its own past representations
        for batch in range(int((N / g) / t)):
            start_node = batch * (g * t)
            for reverse_timestep in range(t):
                timestep = t - reverse_timestep - 1
                for i in range(g):
                    current_node = start_node + timestep * g + i
                    for j in range(max(0, timestep - k_past_self), timestep + 1):
                        past_node = current_node - (timestep - j) * g
                        if past_node >= start_node:
                            edge = (past_node, current_node)
                            edges.add(edge)
                            timesteps[edge] = timestep

        sorted_edges = sorted(edges)
        timestep_values = [timesteps[edge] for edge in sorted_edges]
        return sorted_edges, timestep_values

    def sample_edges(self, edges, timesteps, s):
        """
        Randomly sample s edges from the edge list.

        Args:
            edges: List of (src, dst) edge tuples
            timesteps: List of timesteps for each edge
            s: Number of edges to sample

        Returns:
            sampled_edges, sampled_timesteps
        """
        sampled_indices = random.sample(range(len(edges)), min(s, len(edges)))
        sampled_edges = [edges[i] for i in sampled_indices]
        sampled_timesteps = [timesteps[i] for i in sampled_indices]
        return sampled_edges, sampled_timesteps

    def forward(self, agent_qs, batch, hidden_states=None):
        states = batch["state"]
        obs = batch["obs"]
        hidden_states = obs
        bs = agent_qs.size(0)
        ts = hidden_states.size(1)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # Build fully connected graph within each timestep
        neighbor_table = {i: [] for i in range(bs * ts * self.n_agents)}
        static_edges = set()
        for timestep in range(bs * ts):
            block_start = timestep * self.n_agents
            block_end = block_start + self.n_agents
            for i in range(self.n_agents):
                node = block_start + i
                for neighbor in range(block_start, block_end):
                    if neighbor == node:
                        continue
                    static_edges.add((node, neighbor))

        sorted_static_edges = sorted(static_edges)
        sorted_static_edges = th.tensor(sorted_static_edges).T.to(hidden_states.device)

        # GAT pass: compute attention weights over the static graph
        hidden_states, (edge_index, attention_weights) = self.gat(
            hidden_states, edge_index=sorted_static_edges, return_attention_weights=True
        )

        timestep_per_edge = edge_index[0] // self.n_agents

        # Filter edges: keep top (1 - k_percent) attention edges per timestep
        filtered_edges = []
        for timestep in range(bs * ts):
            t_mask = (timestep_per_edge == timestep)
            if t_mask.sum() == 0:
                continue
            t_mask = t_mask.nonzero(as_tuple=True)[0]
            t_attention = attention_weights[t_mask]
            median_val = th.quantile(t_attention, self.args.k_percent)
            keep_mask = t_attention >= median_val
            t_indices = t_mask[keep_mask.squeeze(1)]
            for i in t_indices:
                filtered_edges.append(edge_index[:, i])

        filtered_edge_index = th.stack(filtered_edges, dim=1)

        # Populate neighbor table from filtered edges
        for src, dst in filtered_edge_index.t().tolist():
            if src != dst:
                neighbor_table[src].append(dst)
                neighbor_table[dst].append(src)

        # Build temporal edges and run TGAT
        edges, edge_timesteps = self.generate_edges_with_reset_timesteps_no_interlinks(
            bs * ts * self.n_agents, self.n_agents, 3, ts, neighbor_table
        )
        sampled_edges, sampled_timesteps = self.sample_edges(
            edges, edge_timesteps, bs * ts * self.n_agents
        )
        sampled_edges = th.tensor(sampled_edges).T
        train_src_l = sampled_edges[0].tolist()
        train_dst_l = sampled_edges[1].tolist()
        train_e_idx_l = list(range(1, sampled_edges.shape[1] + 1))
        train_ts_l = sampled_timesteps

        adj_list = [[] for _ in range(bs * ts * self.n_agents + 1)]
        for src, dst, eidx, tss in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
            adj_list[src].append((dst, eidx, tss))
            adj_list[dst].append((src, eidx, tss))

        ngh_finder = NeighborFinder(adj_list)
        self.tgan.ngh_finder = ngh_finder
        hidden_states = self.tgan(
            n_feat_th=hidden_states,
            src_idx_l=np.array(train_src_l),
            cut_time_l=np.array(train_ts_l),
        )

        # QMIX mixing network
        hidden_states = hidden_states.reshape(-1, self.n_agents, self.hidden_dim)
        hidden_flat = hidden_states.view(bs * ts, -1)
        combined_states = th.cat([states, hidden_flat], dim=-1)

        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(combined_states).view(-1, 1, 1)

        y = th.bmm(hidden, w_final) + v
        q_tot = y.view(bs, -1, 1)
        return q_tot
