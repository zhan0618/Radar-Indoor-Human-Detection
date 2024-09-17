import os
import torch

from utils import *
from trainer_s3 import *

def get_trainer(args,logger):
    if args.dis_loss:
        trainer = DistilationTrainer(args,logger)
    else:
        if args.transfer_loss:
            trainer = DomainAdaptTrainer(args,logger)
        else:
            if args.model in ['baseline','unet','resnet18','resnet34']:
                trainer = BaseTrainer(args,logger)
    return trainer

if __name__ == "__main__":
    set_seed()
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    logger = define_logger(args)
    for key in vars(args):
        logger.info(f"{key}: {vars(args)[key]}")
    if not os.path.exists("saved_model_pth"):
        os.makedirs("saved_model_pth")

    # Define trainer
    trainer = get_trainer(args,logger)

    # Train process
    trainer.train()

    # Visualiza
    #trainer.img_visualization()
    
    # DA result
    print('DA result')
    acc,loss, cm = trainer.test_target()
    n_acc = cm[0,0]/(cm[0,0]+cm[0,1])
    p_acc = cm[1,1]/(cm[1,1]+cm[1,0])
    f1_score = 2*cm[1,1]/(2*cm[1,1]+cm[1,0]+cm[0,1])
    print(f'acc:{acc},positive_acc:{p_acc},negative_acc:{n_acc}, f1_score:{f1_score}')
    logging.info(f'DA result:\n acc {str(acc)}, positive_acc: {p_acc}, negative_acc: {n_acc}, f1_score: {f1_score}')

    # Write to csv
    with open('result.csv','a') as f:
        f.write(f'{args.experiment_index},{acc},{p_acc},{n_acc},{f1_score}\n')







    
