"""
Implementation of Activity Forecasting Training

Shinnosuke Usami, 2019
susami@andrew.cmu.edu
"""

def get_args():

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--algo', '-a', type=int, default=0)
    p.add_argument('--discount', '-d', type=float, default=0.9)
    p.add_argument('--n_epochs', '-e', type=int, default=10)
    p.add_argument('--grid_size', '-g', type=int, default=4)
    p.add_argument('--traj_len', '-l', type=int, default=25)
    p.add_argument('--macro_cell_size', '-m', type=int, default=2)
    p.add_argument('--n_trajs', '-n', type=int, default=5)
    p.add_argument('--learning_rate', '-r', type=float, default=0.01)
    p.add_argument('--wind', '-w', type=float, default=0.1)

    return p.parse_args()


def init_algo(algo_id, n_states, n_actions, discount, trans_prob, feature_mat):
    """Returns the instantiated algorithm with the given settings"""
    if algo_id == 0:
        from algo.maxent_irl import MaxEntIRL
        return MaxEntIRL(n_states, n_actions, discount, trans_prob, feature_mat)

def main():
    args = get_args()

    # create environment

    # feature matrix

    # init_algo

    # recover reward

if __name__ == "__main__":
    main()


