import networkx as nx
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import scipy.sparse as sp


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR models to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def graph_visualization(adj):

    adj = np.array(adj)
    adj = adj - np.eye(10)

    def get_matrix_triad(coo_mathrix, data=False):

        if not sp.isspmatrix_coo(coo_mathrix):
            coo_mathrix = sp.coo_matrix(coo_mathrix)

        temp = np.vstack((coo_mathrix.row, coo_mathrix.col, coo_mathrix.data)).transpose().astype(float)
        return temp.tolist()

    edags = get_matrix_triad(adj)
    G = nx.DiGraph()
    H = nx.path_graph(adj.shape[0])
    G.add_nodes_from(H)

    G.add_weighted_edges_from(edags)
    colors = np.arange(adj.shape[0])
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_color=colors)
    plt.show()
    pass


def simulate_three_domain_var_2(p, T, lag, sd, scale=0.3, is_wrong_structure=True, is_wrong_nonlinear=True,
                                is_wrong_noise=True, is_train=True):

    if is_train:
        np.random.seed(0)
    else:
        np.random.seed(10)

    one_hot_correct_struct = correct_structures = [[1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
                        [0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1]]

    wrong_structures = [[1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1]]

    correct_structures = make_var_stationary(np.asarray(correct_structures))
    wrong_structures = make_var_stationary(np.asarray(wrong_structures))

    burn_in = 100

    domain1_errors = np.random.uniform(low=-1, high=1, size=(p, T + burn_in))
    domain1_X = np.zeros((p, T + burn_in))

    data_list = []

    seq_len = 16
    max_parent = 3

    t = lag

    count = 0

    while t < T + burn_in - seq_len:
        t_env = np.sin(t-1)
        noise = 3 * scale * domain1_errors[:, t-1]

        if t < 208:
            for var_idx in range(p):
                domain1_X[var_idx, t] = sigmoid(
                    np.sum(correct_structures[:, var_idx] * domain1_X[:, (t - lag):t].flatten(order='F')) \
                    + 3 * scale * domain1_errors[var_idx, t - 1] + 0.01 * t_env) + 0.05 * noise[var_idx]

                current_hidden_var = domain1_X[var_idx, t]
                current_observed_var = current_hidden_var * 0.5
                previous_obs_var = domain1_X[:, (t - lag):t].flatten(order='F') * 0.5
                previous_obs_var = previous_obs_var * correct_structures[:, var_idx]
                previous_condition_var = previous_obs_var[previous_obs_var != 0]
                previous_condition_var = list(previous_condition_var)

                if len(previous_condition_var) < max_parent:

                    non_parent_list = np.where(correct_structures[:, var_idx] == 0)[0].tolist()
                    random_parent_list = random.sample(list(non_parent_list), max_parent-len(previous_condition_var))
                    for random_parent_idx in random_parent_list:
                        previous_condition_var.append(domain1_X[random_parent_idx, (t - lag):t][0])

                previous_env_var = t_env
                previous_condition_var = list(previous_condition_var)
                previous_condition_var.append(previous_env_var)
                label = 0
                one_frame_data = [current_observed_var, previous_condition_var, label]
            t += 1
            pass
        else:
            wrong_time_step = random.sample(list(range(1, seq_len)), 1)[0]

            vedio_frame_list = list()
            for vedio_frame_idx in range(seq_len):
                for var_idx in range(p):
                    if vedio_frame_idx == wrong_time_step and is_wrong_structure == True:
                        structure = wrong_structures
                    else:
                        structure = correct_structures

                    if vedio_frame_idx == wrong_time_step and is_train ==False and var_idx == (p-1):
                        print("wrong_time_step: ", wrong_time_step, "wrong_node")

                        structure = wrong_structures
                        domain1_X[var_idx, t] = \
                            (np.sum(structure[:, var_idx] * domain1_X[:, (t - lag):t].flatten(order='F')) \
                                    + 3 * scale * domain1_errors[var_idx, t - 1] + 0.01 * t_env) + 0.5 * noise[var_idx]

                        s = random.randint(-5, 5)

                        print((np.sum(structure[:, var_idx] * domain1_X[:, (t - lag):t].flatten(order='F')) \
                                    + 3 * scale * domain1_errors[var_idx, t - 1] + 0.01 * t_env))

                        current_hidden_var = domain1_X[var_idx, t]
                        current_observed_var = current_hidden_var * 0.5
                        previous_obs_var = domain1_X[:, (t - lag):t].flatten(order='F') * 0.5
                        previous_obs_var = previous_obs_var * structure[:, var_idx]
                        previous_condition_var = previous_obs_var[previous_obs_var != 0]
                        previous_condition_var = list(previous_condition_var)

                        if len(previous_condition_var) < max_parent:

                            non_parent_list = np.where(structure[:, var_idx] == 0)[0].tolist()
                            random_parent_list = random.sample(list(non_parent_list),
                                                               max_parent - len(previous_condition_var))
                            for random_parent_idx in random_parent_list:
                                previous_condition_var.append(domain1_X[random_parent_idx, (t - lag):t][0])

                        previous_env_var = t_env
                        previous_condition_var = list(previous_condition_var)
                        previous_condition_var.append(previous_env_var)
                        label = 1
                        one_frame_data = [current_observed_var, previous_condition_var, label]
                        vedio_frame_list.append(one_frame_data)
                        pass
                    else:
                        count += 1
                        domain1_X[var_idx, t] = \
                            sigmoid(np.sum(structure[:, var_idx] * domain1_X[:, (t - lag):t].flatten(order='F')) \
                                 + 3 * scale * domain1_errors[var_idx, t - 1] + 0.01 * t_env) + 0.5 * noise[var_idx]

                        current_hidden_var = domain1_X[var_idx, t]
                        current_observed_var = current_hidden_var * 0.5
                        previous_obs_var = domain1_X[:, (t - lag):t].flatten(order='F') * 0.5
                        previous_obs_var = previous_obs_var * correct_structures[:, var_idx]
                        previous_condition_var = previous_obs_var[previous_obs_var != 0]
                        previous_condition_var = list(previous_condition_var)
                        if len(previous_condition_var) < max_parent:
                            non_parent_list = np.where(correct_structures[:, var_idx] == 0)[0].tolist()
                            random_parent_list = random.sample(list(non_parent_list),
                                                               max_parent - len(previous_condition_var))
                            for random_parent_idx in random_parent_list:
                                previous_condition_var.append(domain1_X[random_parent_idx, (t - lag):t][0])

                        previous_env_var = t_env
                        previous_condition_var = list(previous_condition_var)
                        previous_condition_var.append(previous_env_var)
                        label = 0
                        one_frame_data = [current_observed_var, previous_condition_var, label]
                        vedio_frame_list.append(one_frame_data)
                t += 1

            data_list.append(vedio_frame_list)

    return data_list


if __name__ == '__main__':

    is_train = False

    data = simulate_three_domain_var_2(p=10, T=20000, lag=1, sd=0.3, is_wrong_structure=False,
                                       is_wrong_nonlinear=False, is_wrong_noise=True, is_train=is_train)

    test_data = simulate_three_domain_var_2(p=10, T=20000, lag=1, sd=0.3, is_wrong_structure=False,
                                       is_wrong_nonlinear=False, is_wrong_noise=True, is_train=True)

    train_data_name = "dataset/train.pkl"
    test_data_name = "dataset/test.pkl"
    train_data_file = open(train_data_name, "wb")
    test_data_file = open(test_data_name, "wb")

    test_data = test_data + data[700:]

    pickle.dump(data[:700], train_data_file)
    pickle.dump(test_data, test_data_file)
