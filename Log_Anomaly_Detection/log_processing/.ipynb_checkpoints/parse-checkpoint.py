import pandas as pd
import re
import os
from collections import OrderedDict
from .logparsers.Spell.Spell import LogParser as SLP
from .logparsers.Drain.Drain import LogParser as DLP
from .logparsers.IPLoM.IPLoM import LogParser as ILP

# PARSE THE DATASET
class LogParser:
    '''
    CLASS FOR PARSING A GIVEN DATASET WITH A GIVEN LOG PARSER
    '''
    def __init__(self, dataset, logparser, clients, parsed_data_dir, raw_data_clients_dir):
        self.dataset = dataset
        self.logparser = logparser
        self.clients = clients
        self.parsed_data_dir = parsed_data_dir
        self.raw_data_clients_dir = raw_data_clients_dir
        self.log_format, self.regex = self._get_log_format_and_regex()

    def _get_log_format_and_regex(self):
        if self.dataset == 'BGL':
            log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
            regex = [r'core\.\d+']
        elif self.dataset == 'HDFS':
            log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
            regex = ["(/[-\w]+)+", "(blk_-?\d+)"]
        else:
            raise ValueError(f'The dataset {self.dataset} is not recognized.')
        return log_format, regex

    def _create_client_directory(self, client_num):
        client_out_dir = self.parsed_data_dir + f'client_data_{client_num}/'
        if not os.path.exists(client_out_dir):
            os.makedirs(client_out_dir)
        return client_out_dir

    def _parse_with_ILP(self, client_data, client_out_dir):
        CT = 0.4
        lowerBound = 0.01
        parser = ILP(
            indir=self.raw_data_clients_dir,
            outdir=client_out_dir,
            log_format=self.log_format,
            CT=CT,
            lowerBound=lowerBound,
            rex=self.regex
        )
        parser.parse(client_data)

    def _parse_with_DLP(self, client_data, client_out_dir):
        depth = 3
        st = 0.3
        parser = DLP(
            indir=self.raw_data_clients_dir,
            outdir=client_out_dir,
            log_format=self.log_format,
            st=st,
            depth=depth,
            rex=self.regex
        )
        parser.parse(client_data)

    def _parse_with_SLP(self, client_data, client_out_dir):
        tau = 0.5
        parser = SLP(
            indir=self.raw_data_clients_dir,
            outdir=client_out_dir,
            log_format=self.log_format,
            tau=tau,
            rex=self.regex
        )
        parser.parse(client_data)

    def parse_logs(self):
        for i in range(self.clients):
            client_data = f'client_{i+1}.log'
            client_out_dir = self._create_client_directory(i+1)

            if self.logparser =='IPLoM':
                print('Parser: IPLoM')
                self._parse_with_ILP(client_data, client_out_dir)

            elif self.logparser == 'Drain':
                print('Parser: Drain')
                self._parse_with_DLP(client_data, client_out_dir)

            elif self.logparser == 'Spell':
                print('Parser: Spell')
                self._parse_with_SLP(client_data, client_out_dir)

            else:
                raise ValueError(f'The log parser {self.logparser} is not recognized.')