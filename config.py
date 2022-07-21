# -*- coding: utf-8 -*-
# @Time    : 2022-03-25 18:33
# @Author  : Zhikang Niu
# @FileName: config.py
# @Software: PyCharm

import argparse
import logging

def get_option(parser= argparse.ArgumentParser()):
    parser.add_argument('--batch_size',type=int,default=32,help="input batch size,default = 64")
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs to train for, default=10')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument("--seed",type = int,default=66,help="random seed")
    parser.add_argument("--show",action="store_true",default=False)
    parser.add_argument("--checkpoint_dir",type=str,default="./checkpoint")
    parser.add_argument("--log_file",type=str,default="./KD_logger.txt")
    parser.add_argument("--GPUS",type=int,default=1)
    args = parser.parse_args()

    if args.show:
        logging.info(f"batch size: {args.batch_size}")
        logging.info(f"GPUS: {args.GPUS}")
        logging.info(f"lr: {args.lr}")


    return args


if __name__ == '__main__':
    ops = get_option()
    print(ops)