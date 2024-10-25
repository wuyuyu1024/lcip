from main_wafhq import load_saved_paper, train_new_model_gan
from main_basic import train_new_model_basic


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load_paper', action='store_true', help='Load the saved model from the paper. This will ignore the arguments setting projection (-p, --projeciton) and inverse projection (-i, --pinv).')
    parser.add_argument('-d', '--dataset', type=str, help='Choose the dataset. The default is wAFHQv2', choices=['mnist', 'fashionmnist', 'w_afhq', 'blob'], default='wAFHQv2')
    parser.add_argument('-p', '--projection', type=str, help='Choose the projection method. The default is umap', choices=['umap', 'tsne', 'mds' ], default='tsne')
    parser.add_argument('-i', '--pinv', type=str, help='Choose the inverse projection method. The default is lcip', choices=['ilamp', 'nninv', 'rbf', 'lcip', 'imds'], default='lcip')
    parser.add_argument('-c', '--clf', action='store_true', help='Train a classifier for decision maps')
    parser.add_argument('-g', '--grid', type=int, help='Gri size (resolution) for decision maps. Default: 100.', default=100)
    args = parser.parse_args()
    # print(args.load_paper)

    if not args.clf:
        clf = None
    else:
        clf = True

    if args.load_paper:
        print('Loading the saved model from the paper')
        load_saved_paper('./models/wAFHQv2_paper', clf=clf)
        
    else:
        print(f'Training a new model on {args.dataset} dataset')
        match args.dataset:
            case 'w_afhq':
                train_new_model_gan(P_name=args.projection, Pinv_name=args.pinv, clf=clf, GRID=args.grid)
            case 'mnist':
                train_new_model_basic(dataset_name='mnist', P_name=args.projection, Pinv_name=args.pinv, clf=clf, GRID=args.grid)
            case 'fashionmnist':
                train_new_model_basic(dataset_name='fashion_mnist', P_name=args.projection, Pinv_name=args.pinv, clf=clf, GRID=args.grid)
            case 'blob':
                train_new_model_basic(dataset_name='blobs', P_name=args.projection, Pinv_name=args.pinv, clf=clf, GRID=args.grid)