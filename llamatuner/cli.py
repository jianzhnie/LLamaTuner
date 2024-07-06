import os
import random
import subprocess
import sys
from enum import Enum, unique

sys.path.append(os.getcwd())
from llamatuner.tuner import launch, run_exp
from llamatuner.utils.env import VERSION, print_env_info
from llamatuner.utils.logger_utils import get_logger
from llamatuner.utils.misc import get_device_count

USAGE = (
    '-' * 70 + '\n' +
    '| Usage:                                                           |\n' +
    '|   llamatuner-cli train -h: train models                          |\n' +
    '-' * 70)

WELCOME = ('-' * 58 + '\n' +
           '| Welcome to LLamaTuner, version {}'.format(VERSION) + ' ' *
           (21 - len(VERSION)) + '|\n|' + ' ' * 56 + '|\n' +
           '| Project page: https://github.com/hiyouga/LLaMA-Factory |\n' +
           '-' * 58)

logger = get_logger(__name__)


@unique
class Command(str, Enum):
    API = 'api'
    CHAT = 'chat'
    ENV = 'env'
    EVAL = 'eval'
    EXPORT = 'export'
    TRAIN = 'train'
    WEBDEMO = 'webchat'
    WEBUI = 'webui'
    VER = 'version'
    HELP = 'help'


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.ENV:
        print_env_info()
    elif command == Command.TRAIN:
        force_torchrun = os.environ.get('FORCE_TORCHRUN',
                                        '0').lower() in ['true', '1']
        if force_torchrun or get_device_count() > 1:
            master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
            master_port = os.environ.get('MASTER_PORT',
                                         str(random.randint(20001, 29999)))
            logger.info('Initializing distributed tasks at: {}:{}'.format(
                master_addr, master_port))
            process = subprocess.run(
                ('torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} '
                 '--master_addr {master_addr} --master_port {master_port} {file_name} {args}'
                 ).format(
                     nnodes=os.environ.get('NNODES', '1'),
                     node_rank=os.environ.get('RANK', '0'),
                     nproc_per_node=os.environ.get('NPROC_PER_NODE',
                                                   str(get_device_count())),
                     master_addr=master_addr,
                     master_port=master_port,
                     file_name=launch.__file__,
                     args=' '.join(sys.argv[1:]),
                 ),
                shell=True,
            )
            sys.exit(process.returncode)
        else:
            run_exp()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError('Unknown command: {}'.format(command))
