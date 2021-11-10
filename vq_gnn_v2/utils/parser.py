import argparse

def parse() :
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--no-second-fc', action='store_false', default=True)
    parser.add_argument('--EMA', action='store_false', default=True)
    parser.add_argument('--split', action='store_false', default=True)

    parser.add_argument('--log-steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--cluster', type=str, default='vq')
    parser.add_argument('--ln-para', action='store_true', default=False)
    parser.add_argument('--kmeans-init', action='store_true')
    parser.add_argument('--kmeans-iter', type=int, default=100)
    parser.add_argument('--dropbranch', type=float, default=0.)
    parser.add_argument('--weight-ahead', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--commitment-cost', type=float, default=0.)
    parser.add_argument('--num-branch', type=int, default=0)
    parser.add_argument('--ce-only', action='store_true')
    parser.add_argument('--sche', action='store_true')
    parser.add_argument('--use-gcn', action='store_true')  # not used

    parser.add_argument('--data-root', type=str, default='/cmlscratch/kong/datasets')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)

    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--test-batch-size', type=int, default=60000)
    parser.add_argument('--num-M', type=int, default=256)
    parser.add_argument('--num-D', type=int, default=4)
    parser.add_argument('--grad-scale', nargs='+', type=float, default=[1, 1])
    parser.add_argument('--act', type=str, default='leaky_gelu')
    parser.add_argument('--bn-flag', action='store_false', default=True)
    parser.add_argument('--warm-up', action='store_false', default=True)
    parser.add_argument('--warm-up-epochs', type=float, default=0)

    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--conv-type', type=str, default='GCN',
                        choices=['GCN', 'SAGE', 'GAT'])
    parser.add_argument('--transformer-flag', action='store_true')
    parser.add_argument('--clip', nargs='+', type=float, default=None)  # w, a
    parser.add_argument('--dataset', type=str, default='arxiv',
                        choices=['arxiv', 'products', 'yelp', 'reddit', 'flickr', 'ppi', 'cluster', 'collab', 'citation2'])
    parser.add_argument('--alpha-dropout-flag', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)


    parser.add_argument('--sampler-type', type=str, default='node',
                        choices=['node', 'edge', 'rw', 'cont', 'cluster'])
    parser.add_argument('--num-parts', type=int, default=1)
    parser.add_argument('--recovery-flag', action='store_false', default=True) # always recover in this version
    parser.add_argument('--walk-length', type=int, default=5)
    parser.add_argument('--cont-sliding-window', type=int, default=1)

    parser.add_argument('--exp-name', type=str, default='test')
    parser.add_argument('--exp', action='store_true')
    parser.add_argument('--exp-tag', type=str, default='exp')

    parser.add_argument('--run-idx', type=int)

    args = parser.parse_args()
    print(args)
    return args