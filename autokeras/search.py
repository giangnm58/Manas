import logging
import math
import os
import queue
import re
import sys
import time
import torch
import numpy
import operator
import torch.multiprocessing as mp

from abc import ABC, abstractmethod
from datetime import datetime
import ast
from autokeras.backend import Backend
from autokeras.bayesian import BayesianOptimizer
from autokeras.constant import Constant

from autokeras.nn.generator import mined_model

from autokeras.utils import pickle_to_file, pickle_from_file, verbose_print, get_system

total = datetime.now()
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.gmeans import gmeans
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES, SIMPLE_SAMPLES

# first_best_model = -1


class Searcher(ABC):
    """The base class to search for neural architectures.
    This class generate new architectures, call the trainer to train it, and update the optimizer.
    Attributes:
        n_classes: Number of classes in the target classification task.
        input_shape: Arbitrary, although all dimensions in the input shaped must be fixed.
            Use the keyword argument `input_shape` (tuple of integers, does not include the batch axis)
            when using this layer as the first layer in a model.
        verbose: Verbosity mode.
        history: A list that stores the performance of model. Each element in it is a dictionary of 'model_id',
            'loss', and 'metric_value'.
        neighbour_history: A list that stores the performance of neighbor of the best model.
            Each element in it is a dictionary of 'model_id', 'loss', and 'metric_value'.
        path: A string. The path to the directory for saving the searcher.
        metric: An instance of the Metric subclasses.
        loss: A function taking two parameters, the predictions and the ground truth.
        generators: A list of generators used to initialize the search.
        model_count: An integer. the total number of neural networks in the current searcher.
        descriptors: A dictionary of all the neural network architectures searched.
        trainer_args: A dictionary. The params for the constructor of ModelTrainer.
        default_model_len: An integer. Number of convolutional layers in the initial architecture.
        default_model_width: An integer. The number of filters in each layer in the initial architecture.
        training_queue: A list of the generated architectures to be trained.
        x_queue: A list of trained architectures not updated to the gpr.
        y_queue: A list of trained architecture performances not updated to the gpr.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss, generators, verbose, num_model,
                 trainer_args=None,
                 default_model_len=None,
                 default_model_width=None,
                 skip_conn=True):
        """Initialize the Searcher.
        Args:
            n_output_node: An integer, the number of classes.
            input_shape: A tuple. e.g. (28, 28, 1).
            path: A string. The path to the directory to save the searcher.
            metric: An instance of the Metric subclasses.
            loss: A function taking two parameters, the predictions and the ground truth.
            generators: A list of generators used to initialize the search.
            verbose: A boolean. Whether to output the intermediate information to stdout.
            trainer_args: A dictionary. The params for the constructor of ModelTrainer.
            default_model_len: An integer. Number of convolutional layers in the initial architecture.
            default_model_width: An integer. The number of filters in each layer in the initial architecture.
        """
        if trainer_args is None:
            trainer_args = {}
        self.n_classes = n_output_node
        self.input_shape = input_shape
        self.verbose = verbose
        self.history = []
        self.neighbour_history = []
        self.path = path
        #self.path = 'ak'
        self.metric = metric
        self.loss = loss
        self.generators = generators
        self.model_count = 0
        self.descriptors = []
        self.trainer_args = trainer_args
        self.default_model_len = default_model_len if default_model_len is not None else Constant.MODEL_LEN
        self.default_model_width = default_model_width if default_model_width is not None else Constant.MODEL_WIDTH
        self.skip_conn = skip_conn

        # giang
        self.num_model = num_model
        self.temp_list = []
        self.count_temp = 0
        self.first_best_model = -1
        self.link_best_model = ''
        self.max_metric = 0
        self.max_loss = 0
        self.best_dnn = -1
        self.CNN = ['conv1d', 'conv2d', 'conv3d']
        self.lin = ['linear']
        self.drop = ['dropout']
        self.conv = []
        self.file_arr = []
        if 'max_iter_num' not in self.trainer_args:
            self.trainer_args['max_iter_num'] = Constant.SEARCH_MAX_ITER

        self.training_queue = []
        self.training_queue_temp = []
        self.x_queue = []
        self.y_queue = []

        logging.basicConfig(filename=os.path.join(self.path, datetime.now().strftime('run_%d_%m_%Y_%H_%M.log')),
                            format='%(asctime)s - %(filename)s - %(message)s', level=logging.DEBUG)

        self._timeout = None

    def load_model_by_id(self, model_id):
        return pickle_from_file(os.path.join(self.path, str(model_id) + '.graph'))

    def load_best_model(self):
        #print(self.path,'bbbbbbbbb')
        #print(self.get_best_model_id(), 'bbbbbbbbb')
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        for item in self.history:
            if item['model_id'] == model_id:
                return item['metric_value']
        return None

    def get_path(self):
        path = [item for item in self.temp_list if item[0] == self.first_best_model]
        return path[0][1]

    def get_best_model_id(self):

        if self.metric.higher_better():
            for item in self.history:
                if item['metric_value'] > self.max_metric:
                    self.max_metric = item['metric_value']
                    self.max_loss = item['loss']
                    self.best_dnn = item['model_id']
                if item['metric_value'] == self.max_metric:
                    if item['loss'] < self.max_loss:
                        self.best_dnn = item['model_id']
            return self.best_dnn
            #return max(self.history, key=lambda x: x['metric_value'])['model_id']
        return min(self.history, key=lambda x: x['metric_value'])['model_id']

    def replace_model(self, graph, model_id):
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))
    def get_meta_info(self):
        return (self.input_shape[2], self.n_classes)
    '''
    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        if self.verbose:
            print('\nInitializing search.')
        #for generator in self.generators:
        graph = Graph(self.input_shape, False)
        output_node_id = 0
        n_nodes_prev_layer = self.input_shape[0]
        for i in range(2):
            output_node_id = graph.add_layer(StubDense(n_nodes_prev_layer, 512), output_node_id)
            output_node_id = graph.add_layer(StubDropout1d(0.2), output_node_id)
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            n_nodes_prev_layer = 512
        output_node_id = graph.add_layer(StubSoftmax(), output_node_id)
        graph.add_layer(StubDense(n_nodes_prev_layer, 10), output_node_id)
        model_id = self.model_count
        print(model_id)
        print(graph.produce_model())
        self.model_count += 1
        self.training_queue.append((graph, -1, model_id))
        self.descriptors.append(graph.extract_descriptor())
        if self.verbose:
            print('Initialization finished.')
    '''
    def distance(self, x1, y1, x2, y2):
        return numpy.math.sqrt(math.pow((x1-x2), 2) + math.pow((y1-y2), 2))

    def most_frequent(self, List):
        counter = 0
        num = List[0]
        for i in List:
            curr_frequency = List.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i
        return num



    def model_clustering(self, meta_info,  dimension_list = {}, path_list = [], count = 0):

        def gmeans_clustering(path_sample):
            # Read sample from file.
            sample = read_sample(path_sample)

            # Create instance of G-Means algorithm.
            # By default algorithm starts search from a single cluster.
            gmeans_instance = gmeans(sample, repeat=5).process()

            # Extract clustering results: clusters and their centers
            clusters = gmeans_instance.get_clusters()

            # Visualize clustering results
            # visualizer = cluster_visualizer()
            # visualizer.append_clusters(clusters, sample)
            # visualizer.show()

            return clusters

        gap_check_0 = 0
        gap_check_1 = 0
        self.temp_list = []
        self.file_arr = []
        self.conv = []

        path_list = []
        temp_array = []
        for r, d, f in os.walk(meta_info):
            for file in f:
                model = open(os.path.join(r, file), "r", encoding="ISO-8859-1")
                candidate_coor = ast.literal_eval(model.readlines()[-1])
                #candidate_distance = self.distance(candidate_coor[0], candidate_coor[1], candidate_coor[2])
                distance = self.distance(self.input_shape[0], self.input_shape[1], candidate_coor[0], candidate_coor[1])
                #dimension_list.update({os.path.join(r, file): [abs(candidate_coor[2] - self.n_classes), distance]})
                temp_array.append([distance, abs(candidate_coor[2] - self.n_classes)])
                dimension_list.update({os.path.join(r, file): [distance, abs(candidate_coor[2] - self.n_classes)]})
                #print(dimension_list)
        sorted_a = sorted(temp_array)
        #print(len(sorted_a), 'xxxxxxxxx')
        string = ''
        for i in sorted_a:
            string += str(i[0]) + " " + str(i[1]) + "\n"
        f = open("dist.data", "w")
        f.write(string)
        f.close()
        clusters = gmeans_clustering("dist.data")
        #print(clusters)

        sorted_d = sorted(dimension_list.items(), key=operator.itemgetter(1))
        count_link = 0
        for i in clusters:
            if 1 in i:
                for link, key in sorted_d:
                    path_list.append(link)
                    self.temp_list.append((count_link, link))
                    count_link += 1
                    if count_link == len(i):
                        break
                break

        #os.remove("dist.data")
        # for link, gap in sorted_d:
        #     #print(gap[0],gap_check_0,'qqqqqqqqqqqqqqqq')
        #     if gap[0] == gap_check_0:
        #         if gap[1] == gap_check_1:
        #             self.temp_list.append((count, link))
        #             #print('11111111111')
        #             count += 1
        #             path_list.append(link)
        #         else:
        #
        #             gap_check_1 = gap[1]
        #             #print(len(path_list), num_model, 'xxxxxxxxxxxxxx')
        #
        #             if len(path_list) < num_model:
        #                 self.temp_list.append((count, link))
        #                 #print('222222222222222222')
        #                 count += 1
        #                 path_list.append(link)
        #             else:
        #                 break
        #     else:
        #         gap_check_0 = gap[0]
        #         if len(path_list) < num_model:
        #             self.temp_list.append((count, link))
        #             #print('333333333333333333')
        #             count += 1
        #             path_list.append(link)
        #         else:
        #             break
            #print(self.temp_list,'whyyyyyyyyyyyyyyyyyyyyyyyy')
        #print(len(path_list), 'mmmmmmmmmmmmmmmmmm')
        #print(self.temp_list, 'whyyyyyyyyyyyyyyyyyyyyyyyy')
        if len(path_list) > 30:
            self.temp_list = []
            for file in path_list:
                fullmodel = []
                linear = []
                droparr = []
                sim = []
                sim_lin = []
                last_layer = 0
                model = open(file, "r", encoding="ISO-8859-1")
                for layer in model:
                    try:
                        dictlayer = ast.literal_eval(layer)
                        if dictlayer['func'].lower() in self.CNN or dictlayer['func'].lower() in self.lin:
                            fullmodel.append(dictlayer)
                        if dictlayer['func'].lower() in self.lin:
                            linear.append(dictlayer)
                        if dictlayer['func'].lower() in self.drop:
                            droparr.append(dictlayer.values())
                    except (SyntaxError, TypeError):
                        pass
                # print(linear,file,'xxxxxxxxxxx')
                for i in range(len(fullmodel)):
                    if isinstance(list(fullmodel[i].values())[1], list):
                        x = list(fullmodel[i].values())[1]
                    else:
                        x = list(fullmodel[i].values())[1]
                    if i == 0:
                        last_layer = x
                    y = list(fullmodel[i].values())[2]
                    sim.append([last_layer, y])
                    last_layer = y

                for j in range(len(linear)):
                    #print(file, list(linear[j].values()))
                    x = list(linear[j].values())[1]
                    if j == 0:
                        last_layer = x
                    y = list(linear[j].values())[2]
                    sim_lin.append([last_layer, y])
                    last_layer = y
                # if sim[0][0] == [1, 28, 28] and sim_lin[len(sim_lin) - 1][1] == 10:
                # if len(sim) == 2 and len(sim_lin) == 2:
                self.conv.append(sim)

            for i in range(len(self.conv)):
                if self.conv[i] != self.most_frequent(self.conv):
                    self.file_arr.append(i)
            self.file_arr.sort(reverse=True)
            #print(self.file_arr)
            for index in range(len(self.file_arr)):
                #print(path_list, len(path_list))
                del path_list[self.file_arr[index]]
            for i in range(len(path_list)):
                self.temp_list.append((i, path_list[i]))



        #print(self.temp_list, 'temp_list')
        #print(path_list, 'xxxxxxxxxxxxxxxxxxxxxxxx')
        count = -1
        for generator in self.generators:
            #print(len(self.generators), 'mmmmmmmmmmmmm')
            if count == len(path_list) - 1:
                #print(count, 'kkkkkkkkkkk')
                break
            # print(count, 'xxxxxxxxxxxx')
            count += 1
            graph = generator(self.n_classes, self.input_shape, path_list[count]). \
                generate(self.default_model_len, self.default_model_width)
            #print(graph.produce_model())
            model_id = self.model_count
            self.model_count += 1
            # self.training_queue.append((graph, -1, model_id))
            self.training_queue_temp.append((graph, -1, model_id, path_list[count]))
            self.descriptors.append(graph.extract_descriptor())
        return path_list
    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""

        if self.verbose:
            print('\nInitializing search.')
        print(self.input_shape[2])
        if self.input_shape[2] == 3:
            self.model_clustering(Constant.model3_10)

        '''
        if self.input_shape[2] == 3 and 10 < self.n_classes <= 100:
            
            self.model_selection(Constant.model3_10)

        if self.input_shape[2] == 3 and 100 < self.n_classes :
            self.model_selection(Constant.model3_1000)
        '''
        if self.input_shape[2] == 1:
            self.model_clustering(Constant.model1_10)
        '''
        if self.input_shape[2] == 1 and 10 < self.n_classes <= 100:
            self.model_selection(Constant.model1_100)

        if self.input_shape[2] == 1 and 100 < self.n_classes:
            self.model_selection(Constant.model1_1000)
        '''
        if self.verbose:
            print('Initialization finished.')

    def init_search_temp(self):

        """Call the generators to generate the initial architectures for the search."""
        trans = 0
        self.first_best_model = self.get_best_model_id()
        #print(self.first_best_model, self.n_classes, self.input_shape, 'kkkkkkkkkkkkk')
        if self.verbose:
            print('\nInitializing search.xxxx')
        self.link_best_model = [item for item in self.temp_list if item[0] == self.first_best_model]
        #print(self.first_best_model,self.temp_list, "whyyyyyyyyyyyyyy")
        graph = self.generators[self.get_best_model_id()](self.n_classes, self.input_shape, self.link_best_model[0][1]). \
            generate(self.default_model_len, self.default_model_width)
        print(graph.produce_model())
        model_id = self.model_count
        self.model_count += 1
        self.training_queue.append((graph, -1, model_id))
        self.descriptors.append(graph.extract_descriptor())

        self.generators.append(mined_model)
        self.generators.append(mined_model)
        self.generators.append(mined_model)


        for i in range(len(self.generators) - 3, len(self.generators)):
            graph = self.generators[i](self.n_classes, self.input_shape, self.link_best_model[0][1], trans). \
                generate(self.default_model_len, self.default_model_width)
            print(graph.produce_model())
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())
            trans += 1

        if self.verbose:
            print('Initialization finished.')

    def search(self, train_data, test_data, timeout=60 * 60 * 24):

        """Run the search loop of training, generating and updating once.
        The function will run the training and generate in parallel.
        Then it will update the controller.
        The training is just pop out a graph from the training_queue and train it.
        The generate will call the self.generate function.
        The update will call the self.update function.
        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """
        current1 = datetime.now()
        self.count_temp += 1
        torch.cuda.empty_cache()
        if self.count_temp == 1:
            # self.init_search()
            self.init_search_temp()

        self._timeout = time.time() + timeout if timeout is not None else sys.maxsize
        self.trainer_args['timeout'] = timeout
        # Start the new process for training.
        graph, other_info, model_id = self.training_queue.pop(0)
        print(graph.produce_model())

        file1 = open(os.path.join(self.path, 'model.txt'), 'a')
        file1.write(str(model_id) + '\n')
        file1.write(str(graph.produce_model()) + '\n')
        file1.close()
        if self.verbose:
            print('\n')
            print('+' + '-' * 46 + '+')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('+' + '-' * 46 + '+')
        # Temporary solution to support GOOGLE Colab
        if get_system() == Constant.SYS_GOOGLE_COLAB:
            # When using Google Colab, use single process for searching and training.
            self.sp_search(graph, other_info, model_id, train_data, test_data)
        else:
            # Use two processes
            self.mp_search(graph, other_info, model_id, train_data, test_data)
        current2 = datetime.now()
        print(current2 - current1)

    def search_temp(self, train_data, test_data):
        """Run the search loop of training, generating and updating once.
        The function will run the training and generate in parallel.
        Then it will update the controller.
        The training is just pop out a graph from the training_queue and train it.
        The generate will call the self.generate function.
        The update will call the self.update function.
        Args:
            train_data: An instance of DataLoader.
            test_data: An instance of Dataloader.
            timeout: An integer, time limit in seconds.
        """
        current1 = datetime.now()

        torch.cuda.empty_cache()
        if not self.history:
            self.init_search()

        # Start the new process for training.
        graph, other_info, model_id, optim_path = self.training_queue_temp.pop(0)
        print(graph.produce_model())

        file1 = open(os.path.join(self.path, 'model.txt'), 'a')
        file1.write(str(model_id) + '\n')
        file1.write(str(graph.produce_model()) + '\n')
        file1.close()
        if self.verbose:
            print('\n')
            print('+' + '-' * 46 + '+')
            print('|' + 'Training model {}'.format(model_id).center(46) + '|')
            print('+' + '-' * 46 + '+')
        # Temporary solution to support GOOGLE Colab
        if get_system() == Constant.SYS_GOOGLE_COLAB:
            # When using Google Colab, use single process for searching and training.
            self.sp_search_temp(graph, other_info, model_id, train_data, test_data, optim_path)
        else:
            # Use two processes
            self.mp_search_temp(graph, other_info, model_id, train_data, test_data, optim_path)
        current2 = datetime.now()
        print(current2 - current1)

    def mp_search_temp(self, graph, other_info, model_id, train_data, test_data, optim_path):
        ctx = mp.get_context()
        q = ctx.Queue()
        p = ctx.Process(target=train, args=(q, graph, train_data, test_data, self.trainer_args,
                                                 self.metric, self.loss, self.verbose, self.path, optim_path))

        p.start()
        metric_value, loss, graph = q.get(block=True)

        if metric_value is not None:
            self.add_model(metric_value, loss, graph, model_id)
            self.update(other_info, model_id, graph, metric_value)

        # terminate and join the subprocess to prevent 1_any resource leak
        p.terminate()
        p.join()

    def sp_search_temp(self, graph, other_info, model_id, train_data, test_data, optim_path):

        metric_value, loss, graph = train(None, graph, train_data, test_data, self.trainer_args,
                                               self.metric, self.loss, self.verbose, self.path, optim_path)
        # Do the search in current thread.

        if metric_value is not None:
            self.add_model(metric_value, loss, graph, model_id)
            self.update(other_info, model_id, graph, metric_value)

    def mp_search(self, graph, other_info, model_id, train_data, test_data):
        ctx = mp.get_context()
        q = ctx.Queue()
        p = ctx.Process(target=train, args=(q, graph, train_data, test_data, self.trainer_args,
                                            self.metric, self.loss, self.verbose, self.path, self.link_best_model[0][1]))
        try:
            p.start()
            search_results = self._search_common(q)
            metric_value, loss, graph = q.get(block=True)
            if time.time() >= self._timeout:
                raise TimeoutError
            if self.verbose and search_results:
                for (generated_graph, generated_other_info, new_model_id) in search_results:
                    verbose_print(generated_other_info, generated_graph, new_model_id)

            if metric_value is not None:
                self.add_model(metric_value, loss, graph, model_id)
                self.update(other_info, model_id, graph, metric_value)

        except (TimeoutError, queue.Empty) as e:
            raise TimeoutError from e
        finally:
            # terminate and join the subprocess to prevent 1_any resource leak
            p.terminate()
            p.join()

    def sp_search(self, graph, other_info, model_id, train_data, test_data):
        try:
            metric_value, loss, graph = train(None, graph, train_data, test_data, self.trainer_args,
                                              self.metric, self.loss, self.verbose, self.path, self.link_best_model[0][1])
            # Do the search in current thread.
            search_results = self._search_common()
            if self.verbose and search_results:
                for (generated_graph, generated_other_info, new_model_id) in search_results:
                    verbose_print(generated_other_info, generated_graph, new_model_id)

            if metric_value is not None:
                self.add_model(metric_value, loss, graph, model_id)
                self.update(other_info, model_id, graph, metric_value)

        except TimeoutError as e:
            raise TimeoutError from e

    def _search_common(self, mp_queue=None):
        search_results = []
        if not self.training_queue:
            results = self.generate(mp_queue)
            for (generated_graph, generated_other_info) in results:
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append((generated_graph, generated_other_info, new_model_id))
                self.descriptors.append(generated_graph.extract_descriptor())
                search_results.append((generated_graph, generated_other_info, new_model_id))
            self.neighbour_history = []

        return search_results

    @abstractmethod
    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.
        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.
        Returns:
            list of 2-element tuples: generated_graph and other_info,
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together
            with the architecture.
        """
        pass

    @abstractmethod
    def update(self, other_info, model_id, graph, metric_value):
        """ Update the controller with evaluation result of a neural architecture.
        Args:
            other_info: Anything. In the case of default bayesian searcher, it is the father ID in the search tree.
            model_id: An integer.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
        """
        pass

    def add_model(self, metric_value, loss, graph, model_id):
        """Append the information of evaluated architecture to history."""

        if self.verbose:
            print('\nSaving model.')

        graph.clear_operation_history()
        pickle_to_file(graph, os.path.join(self.path, str(model_id) + '.graph'))

        ret = {'model_id': model_id, 'loss': loss, 'metric_value': metric_value}
        self.neighbour_history.append(ret)
        self.history.append(ret)

        # Update best_model text file
        current_hold = datetime.now()
        time_block = current_hold - total
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, 'best_model.txt'), 'a')
            # file = open(os.path.join(self.path, 'best_model.txt'), 'w')
            print(model_id)
            file.write(
                'best model: ' + str(model_id) + ' loss: ' + str(loss) + ' metric_value: ' + str(
                    metric_value) + ' time_block: ' + str(time_block) + '\n')
            file.close()
        if self.verbose:
            idx = ['model_id', 'loss', 'metric_value']
            header = ['Model ID', 'Loss', 'Metric Value']
            line = '|'.join(x.center(24) for x in header)
            print('+' + '-' * len(line) + '+')
            print('|' + line + '|')

            if self.history:
                r = self.history[-1]
                print('+' + '-' * len(line) + '+')
                line = '|'.join(str(r[x]).center(24) for x in idx)
                print('|' + line + '|')
            print('+' + '-' * len(line) + '+')
        file1 = open(os.path.join(self.path, 'info.txt'), 'a')
        file1.write('best model: ' + str(model_id) + ' loss: ' + str(loss) + ' metric_value: ' + str(
            metric_value) + ' time_block: ' + str(time_block) + '\n')
        file1.close()
        return ret


class BayesianSearcher(Searcher):
    """ Class to search for neural architectures using Bayesian search strategy.
    Attribute:
        optimizer: An instance of BayesianOptimizer.
        t_min: A float. The minimum temperature during simulated annealing.
    """

    def __init__(self, n_output_node, input_shape, path, metric, loss,
                 generators, verbose, num_model, trainer_args=None,
                 default_model_len=None, default_model_width=None,
                 t_min=None, skip_conn=True):
        super(BayesianSearcher, self).__init__(n_output_node, input_shape,
                                               path, metric, loss,
                                               generators, verbose, num_model,
                                               trainer_args,
                                               default_model_len,
                                               default_model_width,
                                               skip_conn)
        if t_min is None:
            t_min = Constant.T_MIN
        self.optimizer = BayesianOptimizer(self, t_min, metric, skip_conn=self.skip_conn)

    def generate(self, multiprocessing_queue):
        """Generate the next neural architecture.
        Args:
            multiprocessing_queue: the Queue for multiprocessing return value.
        Returns:
            list of 2-element tuples: generated_graph and other_info,
            for bayesian searcher the length of list is 1.
            generated_graph: An instance of Graph.
            other_info: Anything to be saved in the training queue together with the architecture.
        """
        remaining_time = self._timeout - time.time()
        generated_graph, new_father_id = self.optimizer.generate(self.descriptors,
                                                                 remaining_time, multiprocessing_queue)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](self.n_classes, self.input_shape). \
                generate(self.default_model_len, self.default_model_width)

        return [(generated_graph, new_father_id)]

    def update(self, other_info, model_id, graph, metric_value):
        """ Update the controller with evaluation result of a neural architecture.
        Args:
            other_info: Anything. In our case it is the father ID in the search tree.
            model_id: An integer.
            graph: An instance of Graph. The trained neural architecture.
            metric_value: The final evaluated metric value.
        """
        father_id = other_info
        self.optimizer.fit([graph.extract_descriptor()], [metric_value])
        self.optimizer.add_child(father_id, model_id)


def train(q, graph, train_data, test_data, trainer_args, metric, loss, verbose, path, link_best_model=''):
    """Train the neural architecture."""
    lr = None
    momentum = None
    weight_decay = None
    optim = None
    #print(link_best_model, 'aaaaaaaaaaaa')
    try:
        model = graph.produce_model()
        opt = open(link_best_model, 'r', encoding="ISO-8859-1")
        #print(opt.readlines()[-2], 'mmmmmmmmmm')
        optimizer = []
        #optpara = ['Adadelta', 'Adagrad', 'Adam', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD',
                   #'add_argument']
        optpara = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        try:
            line = ast.literal_eval(opt.readlines()[-2])
            if line['func'] == 'compile':
                if 'optimizer' in line:
                    if line['optimizer'].lower() == 'sgd':
                        optim = 'sgd'
                    if line['optimizer'].lower() == 'adam':
                        optim = 'adam'
                    if line['optimizer'].lower() == 'adadelta':
                        optim = 'adadelta'
                    if line['optimizer'].lower() == 'rmsprop':
                        optim = 'rmsprop'
            if line['func'].lower() == 'sgd':
                optim = 'sgd'
                if 'lr' in line:
                    lr = line['lr']
                if 'decay' in line:
                    weight_decay = line['decay']
                if 'momentum' in line:
                    momentum = line['momentum']
            if line['func'].lower() == 'adam':
                optim = 'adam'
                if 'lr' in line:
                    lr = line['lr']
                if 'decay' in line:
                    weight_decay = line['decay']

            if line['func'].lower() == 'adadelta':
                optim = 'adadelta'
                if 'lr' in line:
                    lr = line['lr']
                if 'decay' in line:
                    weight_decay = line['decay']

            if line['func'].lower() == 'rmsprop':
                optim = 'rmsprop'
                if 'lr' in line:
                    lr = line['lr']
                if 'decay' in line:
                    weight_decay = line['decay']
                if 'momentum' in line:
                    momentum = line['momentum']

        except SyntaxError:
            pass
        '''
        for i in range(len(optimizer)):
            for key in optimizer[i]:
                if key == 'lr':
                    lr = optimizer[i][key]
                if key == 'momentum':
                    momentum = optimizer[i][key]
                if key == 'weight_decay':
                    weight_decay = optimizer[i][key]
                if 'arg1' in optimizer[i]:
                    if optimizer[i]['arg1'] == '--lr':
                        lr = optimizer[i]['default']
                    if optimizer[i]['arg1'] == '--momentum':
                        momentum = optimizer[i]['default']
                    if optimizer[i]['arg1'] == '--weight_decay':
                        weight_decay = optimizer[i]['default']
        '''
        loss, metric_value = Backend.get_model_trainer(model=model,
                                                       path=path,
                                                       train_data=train_data,
                                                       test_data=test_data,
                                                       metric=metric,
                                                       loss_function=loss,
                                                       verbose=verbose).train_model(optim=optim, lr=lr, momentum=momentum,
                                                                                    weight_decay=weight_decay,
                                                                                    **trainer_args)
        model.set_weight_to_graph()
        if q:
            q.put((metric_value, loss, model.graph))
        return metric_value, loss, model.graph
    except RuntimeError as e:
        if not re.search('out of memory', str(e)):
            print('\nincorrect add and concat. Discontinuing training this model to search for other models.')
            #q.put((None, None, None))
            #raise e
        if verbose:
            print('\nCurrent model size is too big. Discontinuing training this model to search for other models.')
        Constant.MAX_MODEL_SIZE = graph.size() - 1
        if q:
            q.put((None, None, None))




        return None, None, None
    except TimeoutError as exp:
        logging.warning("TimeoutError occurred at train() : {0}".format(str(exp)))
        if q:
            q.put((None, None, None))
        return None, None, None
    except Exception as exp:
        logging.warning("Exception occurred at train() : {0}".format(str(exp)))
        if verbose:
            print("Exception occurred at train() : {0}".format(str(exp)))
        if q:
            q.put((None, None, None))
        return None, None, None


'''
def train_temp(q, graph, train_data, test_data, trainer_args, metric, loss, verbose, path, optim_path):
    lr = None
    momentum = None
    weight_decay = None
    optim = None
    print(optim_path, 'aaaaaaaaaaaa')
    try:
        model = graph.produce_model()
        opt = open(optim_path,'r', encoding="ISO-8859-1")
        optimizer = []
        #optpara = ['Adadelta', 'Adagrad', 'Adam', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD',
                   #'add_argument']
        optpara = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        for line in opt:
            try:
                line = ast.literal_eval(line)
                if line['func'] == 'compile':
                    if line['optimizer'].lower() == 'sgd':
                        optim = 'sgd'
                    if line['optimizer'].lower() == 'adam':
                        optim = 'adam'
                if line['func'].lower() == 'sgd':
                    optim = 'sgd'
                    if 'lr' in line:
                        lr = line['lr']
                    if 'decay' in line:
                        weight_decay = line['decay']
                    if 'momentum' in line:
                        momentum = line['momentum']
                if line['func'].lower() == 'adam':
                    optim = 'adam'
                    if 'lr' in line:
                        lr = line['lr']
                    if 'decay' in line:
                        weight_decay = line['decay']
            except SyntaxError:
                pass

        loss, metric_value = Backend.get_model_trainer(model=model,
                                                       path=path,
                                                       train_data=train_data,
                                                       test_data=test_data,
                                                       metric=metric,
                                                       loss_function=loss,
                                                       verbose=verbose).train_model(optim=optim, lr=lr, momentum=momentum,
                                                                                    weight_decay=weight_decay,
                                                                                    **trainer_args)
        model.set_weight_to_graph()
        if q:
            q.put((metric_value, loss, model.graph))
        return metric_value, loss, model.graph
    except RuntimeError as e:
        if not re.search('out of memory', str(e)):
            raise e
        if verbose:
            print('\nCurrent model size is too big. Discontinuing training this model to search for other models.')
        Constant.MAX_MODEL_SIZE = graph.size() - 1
        if q:
            q.put((None, None, None))
        return None, None, None
    except TimeoutError as exp:
        logging.warning("TimeoutError occurred at train() : {0}".format(str(exp)))
        if q:
            q.put((None, None, None))
        return None, None, None
    except Exception as exp:
        logging.warning("Exception occurred at train() : {0}".format(str(exp)))
        if verbose:
            print("Exception occurred at train() : {0}".format(str(exp)))
        if q:
            q.put((None, None, None))
        return None, None, None
'''