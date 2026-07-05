import os
import argparse
import torch

from model.ViewFusionRegressor import ViewFusionRegressor
from trainer.ViewFusionRegressorTrainer import ViewFusionRegressorTrainer
from util import getAllTransforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViewFusionRegressor.')

    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--model_family', type=str, default='efficientnet')
    parser.add_argument('--backbone_type', type=str, default='b0')
    parser.add_argument('--features_size', type=int, default=256)

    parser.add_argument('--grade_type', type=str, required=True)

    parser.add_argument('--tabular_path', type=str, default='data/label/label.xlsx')
    parser.add_argument('--photo_path', type=str, default='data/photo/*.jpg')

    parser.add_argument(
        '--internal_lst',
        nargs='+',
        default=['S02', 'S03', 'S05', 'S06', 'S08', 'S10', 'S11']
    )
    parser.add_argument(
        '--external_lst',
        nargs='*',
        default=['S01', 'S09']
    )

    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--start_fold', type=int, default=0)

    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--drop_last', action='store_true', default=True)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--resize_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)

    parser.add_argument('--split_path', type=str, default=None)

    parser.add_argument('--start_factor', type=float, default=0.1)
    parser.add_argument('--end_factor', type=float, default=1.0)
    parser.add_argument('--total_iters', type=int, default=5)

    parser.add_argument('--T_0', type=int, default=15)
    parser.add_argument('--T_mult', type=int, default=2)
    parser.add_argument('--eta_min', type=float, default=1e-7)

    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--stable_epochs', type=int, default=5)
    parser.add_argument('--decay_epochs', type=int, default=35)
    parser.add_argument('--max_lambda', type=float, default=0.1)
    parser.add_argument('--min_lambda', type=float, default=0.001)

    parser.add_argument('--result_root', type=str, default='file/ViewFusionRegressor')

    return parser.parse_args()


def save_default_model_if_needed(model, default_model_path, device):
    os.makedirs(os.path.dirname(default_model_path), exist_ok=True)

    if os.path.exists(default_model_path):
        print(f'[*] Found existing default model: {default_model_path}')
        return

    torch.save(
        {
            'model_state': model.state_dict(),
            'device': str(device)
        },
        default_model_path
    )

    print(f'[*] Default model saved to: {default_model_path}')


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    name = f'{args.model_family}_{args.backbone_type}'

    save_model_path = os.path.join(
        args.result_root,
        name,
        'model',
        f'ViewFusionRegressor_{name}_{args.grade_type}_'
    )

    default_model_path = os.path.join(
        args.result_root,
        name,
        'model',
        f'ViewFusionRegressor_{name}_{args.grade_type}_default.pt'
    )

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    transform_1, transform_2 = getAllTransforms(
        resize_size=args.resize_size,
        crop_size=args.crop_size
    )

    model = ViewFusionRegressor(
        args.grade_type,
        args.model_family,
        args.backbone_type,
        args.features_size
    ).to(device)

    save_default_model_if_needed(
        model,
        default_model_path,
        device
    )

    trainer = ViewFusionRegressorTrainer(
        model=model,
        load_model_path=default_model_path,
        save_model_path=save_model_path,
        lr=args.lr,
        tabular_path=args.tabular_path,
        photo_path=args.photo_path,
        grade_type=args.grade_type,
        num_folds=args.num_folds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        drop_last=args.drop_last,
        transform_1=transform_1,
        transform_2=transform_2,
        internal_lst=args.internal_lst,
        external_lst=args.external_lst,
        start_fold=args.start_fold,
        split_path=args.split_path,
        start_factor=args.start_factor,
        end_factor=args.end_factor,
        total_iters=args.total_iters,
        T_0=args.T_0,
        T_mult=args.T_mult,
        eta_min=args.eta_min,
        warmup_epochs=args.warmup_epochs,
        stable_epochs=args.stable_epochs,
        decay_epochs=args.decay_epochs,
        max_lambda=args.max_lambda,
        min_lambda=args.min_lambda
    )

    trainer.train_evaluate()


if __name__ == '__main__':
    main()