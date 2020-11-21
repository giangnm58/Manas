'''
Creates a control flow graph (cfg)
'''

from clone.model_mining.traversers.astfulltraverser import AstFullTraverser
# from database_creation.repo_download import repo_download
# from database_creation.model_collection import collect_model
# from database_creation.file_filter import ModelMining
from clone.model_mining.database_creation.constant import Constant
import ast
from pprint import pprint


# from graphviz import Digraph


class Block():
    ''' A basic control flow block.
    It has one entry point and several possible exit points.
    Note that the next_block is not necessarily an exit.
    '''

    # Block tags
    NORMAL = 0
    LOOP_HEADER = 1

    def __init__(self):
        # The next block along the function
        self.next_block = None
        self.has_return = False
        # Holds the statements in this block
        self.start_line_no = 0
        self.statements = []
        self.exit_blocks = []
        # Use to indicate whether the block has been visited. Used for printing
        self.marked = False
        # Used to describe special blocks
        self.tag = Block.NORMAL
        # Block which have been absorbed into this one
        self.dependents = []

    def copy_dict(self, copy_to):
        ''' Keep the name bindings but copy the class instances.
            Both bindings now point to the same variables.
            This function is used to simulate C pointers.
            TODO: Find a more elegant way of achieving this. '''
        for dependent in self.dependents:
            dependent.__dict__ = copy_to.__dict__
        self.__dict__ = copy_to.__dict__
        copy_to.dependents = self.dependents + [self]


# These are frame blocks.
# Idea for these are from PyPy
F_BLOCK_LOOP = 0
F_BLOCK_EXCEPT = 1
F_BLOCK_FINALLY = 2
F_BLOCK_FINALLY_END = 3
method_keyword = []
line_number = []
variable = []
dpg = []

'''Added by Giang'''
Assign_val = []
Assign_ID = []
Assign_dict = {}
Model_dict = {}
Layer_list = []
detect_assign = 0
Method_dict = {}
arg_count = 0
layer_count = 0
val_arr1 = []
val_arr2 = []
arr = []

class ControlFlowGraph(AstFullTraverser):
    def __init__(self, apilist, saver):
        self.current_block = None
        # Used to hold how control flow is nested (e.g. if inside of a for)
        self.frame_blocks = []
        self.current_line_num = 0
        self.apis = apilist
        self.saver = saver

    def parse_ast(self, source_ast):
        self.run(source_ast)
        return source_ast

    def parse_file(self, file_path):
        source_ast = self.file_to_ast(file_path)
        return self.parse_ast(source_ast)

    def file_to_ast(self, file_path):
        s = self.get_source(file_path)
        try:
            return ast.parse(s, filename=file_path, mode='exec')
        except ValueError:
            pass

    def get_source(self, fn):
        ''' Return the entire contents of the file whose name is given.
            Almost most entirely copied from stc. '''
        try:
            f = open(fn, encoding="ISO-8859-1")
            s = f.read()
            f.close()
            return s
        except IOError:
            return ''

    def push_frame_block(self, kind, block):
        self.frame_blocks.append((kind, block))

    def pop_frame_block(self, kind, block):
        actual_kind, old_block = self.frame_blocks.pop()
        assert actual_kind == kind and old_block is block, \
            "mismatched frame blocks"

    def is_empty_block(self, candidate_block):
        return not candidate_block.statements

    def check_child_exits(self, candidate_block, after_control_block):
        ''' After if and loop blocks an after_if/loop block is created. If the
            if/loop blocks are the last in a straight line block of statements
            then the after blocks will be empty. All body/then/else exits will
            point to this block. If it is empty then swap for the given block.
            If it is not then set that block's exit as the given block. '''
        if candidate_block.has_return:
            # If the block has a return exit then can not be given another here
            return
        if self.is_empty_block(candidate_block):
            # candidate_block and after_control_block now point to the same
            # variables. They are now the same instance.
            candidate_block.copy_dict(after_control_block)
            return
        # This is needed to avoid two "Exits" appearing for the return or yield
        # at the end of a function.
        if not after_control_block in candidate_block.exit_blocks:
            candidate_block.exit_blocks.append(after_control_block)

    def add_to_block(self, node):
        ''' We want every try statement to be in its own block. '''
        if not self.current_block:
            return
        # We only want the 'top level' statements
        # print(node.__class__.__name__, "xxx")
        try:
            if self.current_line_num >= node.lineno:
                return
        except AttributeError:
            pass
            # Special cases - test must be in its own block
        if isinstance(node, ast.While) or isinstance(node, ast.For):
            if not self.is_empty_block(self.current_block):
                test_block = self.new_block()
                self.current_block.exit_blocks.append(test_block)
                self.use_next_block(test_block)
        try:
            self.current_line_num = node.lineno
        except  AttributeError:
            pass
        for f_block_type, f_block in reversed(self.frame_blocks):
            if f_block_type == F_BLOCK_EXCEPT:
                # Statement is in a try - set exits to next statement and
                # excepts
                self.current_block.statements.append(node)
                for handler in f_block:
                    self.current_block.exit_blocks.append(handler)
                # Special case
                if isinstance(node, ast.While) or isinstance(node, ast.For):
                    break
                next_statement_block = self.new_block()
                self.current_block.exit_blocks.append(next_statement_block)
                self.use_next_block(next_statement_block)
                break
        else:
            self.current_block.statements.append(node)

    def run(self, root):
        self.visit(root)

    def new_block(self):
        ''' From pypy. '''
        return Block()

    def use_block(self, block):
        ''' From pypy. '''
        self.current_block = block

    def empty_block(self, block):
        return not block.statements

    def use_next_block(self, block=None):
        """Set this block as the next_block for the last and use it.
           From pypy """
        if block is None:
            block = self.new_block()
        self.current_block.next_block = block
        self.use_block(block)
        return block

    def add_to_exits(self, source, dest):
        source.exit_blocks.append(dest)

    def check_block_num(self, node, method_lineno=[]):
        ''' Used for display purposes only. Each block is labelled with the
            line number of the the first statement in the block. '''

        if not self.current_block:
            return
        # print(self.current_block.start_line_no, "y")
        if node.__class__.__name__ == "Call":
            method_lineno.append(self.current_block.start_line_no)
        if not self.current_block.start_line_no:
            try:
                self.current_block.start_line_no = node.lineno
            except AttributeError:
                pass
            # print(self.current_block.start_line_no)
            # print(node.lineno)

    def check_has_return(self):
        return self.current_block and self.current_block.has_return



    '''
    def do_FunctionDef(self, node):
        block = self.new_block()
        self.use_block(block)
        node.initial_block = block
        self.exit_block = self.new_block()
        # Special case
        self.exit_block.start_line_no = "Exit"
        for z in node.body:
            self.visit(z)
        # Here there's a chance that the last block already points the exit.
        # Such as yields and returns
        for e in self.current_block.exit_blocks:
            if e.start_line_no == "Exit":
                return
        else:
            self.check_child_exits(self.current_block, self.exit_block)
    '''

    def do_Import(self, node):
        block = self.new_block()
        self.use_block(block)
        node.initial_block = block
        self.exit_block = self.new_block()
        # Special case
        self.exit_block.start_line_no = "Exit"
        # Here there's a chance that the last block already points the exit.
        # Such as yields and returns
        for e in self.current_block.exit_blocks:
            if e.start_line_no == "Exit":
                return
        else:
            self.check_child_exits(self.current_block, self.exit_block)

    def do_ImportFrom(self, node):
        block = self.new_block()
        self.use_block(block)
        node.initial_block = block
        self.exit_block = self.new_block()
        # Special case
        self.exit_block.start_line_no = "Exit"
        # Here there's a chance that the last block already points the exit.
        # Such as yields and returns
        for e in self.current_block.exit_blocks:
            if e.start_line_no == "Exit":
                return
        else:
            self.check_child_exits(self.current_block, self.exit_block)

    def visit(self, node):
        '''Visit a single node. Callers are responsible for visiting children.'''
        if self.check_has_return():
            return

        if node.__class__.__name__ != "keyword":
            self.check_block_num(node)
            self.add_to_block(node)
            # print(node,'kkkk')

        method = getattr(self, 'do_' + node.__class__.__name__)
        # print('do_' + node.__class__.__name__)
        return method(node)


    #need more modifications
    def value_recursive(self, vals, value = None):
        if isinstance(vals, list):
            for val in vals:
                if not isinstance(list(val.__dict__.values())[0], list):
                    arr.append(list(val.__dict__.values())[0])
                value = self.value_recursive(list(val.__dict__.values())[0])
            #print(arr,'kkkkkkkkk')
            val_arr1.append(arr.copy())
            arr.clear()
        if not isinstance(vals, list):
            val_arr1.append(vals)
        return value


    def do_Assign(self, node):
        var = None
        val = None
        #print(node.__dict__, 'Assign')
        #print(node.value.__dict__, 'Assign')
        '''
        for element in node.__dict__.values():
            try:
                if isinstance(element, list):
                    for id in element:
                        var = list(id.__dict__.values())[0]
                        print(list(id.__dict__.values())[0], 'lll')
                else:
                    vals = list(element.__dict__.values())[0]
                    val = self.value_recursive(vals)
                    print(val_arr1, 'lolllllllll')
                Assign_dict[var] = val
            except AttributeError:
                pass
        '''
        block = self.current_block
        next_block = self.new_block()
        self.add_to_exits(block, next_block)
        for z in node.targets:
            self.visit(z)
        self.visit(node.value)
        block.next = next_block
        self.use_block(next_block)

        if 'id' in node.targets[0].__dict__ and 'n' in node.value.__dict__:
            Assign_dict[node.targets[0].__dict__['id']] = node.value.__dict__['n']
        if 'id' in node.targets[0].__dict__ and 's' in node.value.__dict__:
            Assign_dict[node.targets[0].__dict__['id']] = node.value.__dict__['s']
        if 'id' in node.targets[0].__dict__ and 'elts' in node.value.__dict__:
            Assign_dict[node.targets[0].__dict__['id']] = []
            for i in node.value.__dict__['elts']:
                #print(i.__dict__, 'aaaaaaaaaaaa')
                if 'n' in i.__dict__:
                    Assign_dict[node.targets[0].__dict__['id']] += [i.__dict__['n']]
                if 's' in i.__dict__:
                    Assign_dict[node.targets[0].__dict__['id']] += [i.__dict__['s']]
                if 'id' in i.__dict__:
                    #print(str(i.__dict__['id']))
                    if str(i.__dict__['id']) in Assign_dict:
                        Assign_dict[node.targets[0].__dict__['id']] += [Assign_dict[str(i.__dict__['id'])]]

        # if 'elts' in node.targets[0].__dict__:
        # print(node.targets[0].__dict__['elts'][0].__dict__['id'], 'xxxxxxxxxxxxxxxx')

        if 'elts' in node.targets[0].__dict__ and 'elts' in node.value.__dict__:
            for i in range(len(node.value.__dict__['elts'])):
                if 'n' in node.value.__dict__['elts'][i].__dict__:
                    try:
                        Assign_dict[node.targets[0].__dict__['elts'][i].__dict__['id']] = \
                            node.value.__dict__['elts'][i].__dict__['n']
                    except KeyError:
                        pass
                if 's' in node.value.__dict__['elts'][i].__dict__:
                    Assign_dict[node.targets[0].__dict__['elts'][i].__dict__['id']] = \
                        node.value.__dict__['elts'][i].__dict__['s']
                if 'elts' in node.value.__dict__['elts'][i].__dict__:
                    #print(node.value.__dict__['elts'][i].__dict__,'aaaaaaa')
                    try:
                        Assign_dict[node.targets[0].__dict__['elts'][i].__dict__['id']] = \
                            node.value.__dict__['elts'][i].__dict__['elts']
                    except KeyError:
                        pass

        #print(Assign_dict, 'mmmmmmmmmmmm')

    def do_Tuple(self, node):
        # print(node.__dict__, 'Tuple')
        block = self.current_block
        next_block = self.new_block()
        self.add_to_exits(block, next_block)
        for z in node.elts:
            self.visit(z)
            # self.visit(node.ctx)
        block.next = next_block
        self.use_block(next_block)

    def do_List(self, node):
        # print(node.__dict__, 'List')
        block = self.current_block
        next_block = self.new_block()
        self.add_to_exits(block, next_block)
        for z in node.elts:
            self.visit(z)
            # self.visit(node.ctx)
        block.next = next_block
        self.use_block(next_block)

    def do_Num(self, node):
        # print(node.__dict__, 'Num')
        pass

    def do_Str(self, node):
        # print(node.__dict__, 'Str')
        pass

    def do_Name(self, node):
        global arg_count
        # print(node.__dict__, 'Name')

        '''
        if detect_assign == 0:
            Method_dict["arg" + str(arg_count)] = Assign_dict[node.__dict__['id']]
            arg_count += 1
        '''
        # print(Assign_dict)
        # print(Method_dict)

        # print(Method_dict, "method dict")
        variable.append(node.__dict__['id'])
        if node.__dict__['id'] in variable:
            dpg.append(node.__dict__['lineno'])
        block = self.current_block
        next_block = self.new_block()
        self.add_to_exits(block, next_block)
        # self.visit(node.ctx)
        block.next = next_block
        self.use_block(next_block)

    def do_keyword(self, node):
        global detect_assign
        # print(node.__dict__, 'keyword')
        # print(node.value.__dict__, 'keyword')
        # print(node.arg, 'keyword')

        arg = node.__dict__['arg']
        detect_assign += 1
        if 's' in node.value.__dict__:
            Method_dict[str(arg)] = node.value.__dict__['s']
        if 'n' in node.value.__dict__:
            Method_dict[str(arg)] = node.value.__dict__['n']
        if 'id' in node.value.__dict__:
            if (str(arg) in Method_dict):
                Method_dict[str(arg)] = Assign_dict[node.value.__dict__['id']]
        # print(Method_dict, "method dict")
        self.visit(node.value)

    def do_NameConstant(self, node):
        pass

    def do_Expr(self, node):
        # print(node.__dict__, 'Expr')
        # print(node.value.__dict__, 'Expr')
        # print(node.value.func.value.__dict__, 'Expr')
        self.visit(node.value)

    def do_ClassDef(self, node):
        global layer_count

        # print(node.__dict__['name'])
        # f = open("database_creation\classmodel\classes" + str(layer_count) + ".txt", "a", encoding="ISO-8859-1")
        if self.apis == []:
            f = open("tempfiles\\classes.txt", "a", encoding="ISO-8859-1")
            # print('fadsfadsfads')
            f.write(str(node.__dict__['name']) + '\n')
            f.close()
    def do_FunctionDef(self, node):
        #print(node.__dict__, 'Function_def')
        #print(node.args.__dict__, 'Function_def')
        #print(node.args.defaults[0].__dict__['n'], 'Function_def')
        #print(node.args.defaults[0].__dict__, 'Function_def')
        count = 0
        gap = len(node.args.args) - len(node.args.defaults)
        for i in range(gap, len(node.args.args)):
            if len(node.args.defaults) > 0:
                if 'n' in node.args.defaults[count].__dict__:
                    Assign_dict[str(node.args.args[i].__dict__['arg'])] = node.args.defaults[count].__dict__['n']
                    count += 1
        def_block = self.current_block
        exit_def_block = self.new_block()
        self.add_to_exits(def_block, exit_def_block)
        for z in node.body:
            self.visit(z)
        def_block.next = exit_def_block
        self.use_block(exit_def_block)
    def do_Call(self, node):
        #print(node.func.__dict__, 'Call')
        #print(node.args[0].__dict__, 'Call')
        global layer_count
        Method_dict.clear()
        count = 0
        # print(node.func.__dict__)
        block = self.current_block

        next_block = self.new_block()

        self.add_to_exits(block, next_block)

        for z in node.args:
            self.visit(z)

        for z in node.keywords:
            self.visit(z)

        # for torch only
        if 'attr' in node.func.__dict__:
            if node.func.__dict__['attr'] in self.apis:
                Model_dict['func'] = node.func.__dict__['attr']

                for i in range(len(node.args)):
                    count += 1
                    if 'id' in node.args[i].__dict__:
                        if node.args[i].__dict__['id'] in Assign_dict:
                            Model_dict[node.args[i].__dict__['id']] = Assign_dict[node.args[i].__dict__['id']]
                            # print(node.args[i].__dict__['id'])
                    if 's' in node.args[i].__dict__:
                        Model_dict['arg' + str(count)] = node.args[i].__dict__['s']
                    if 'n' in node.args[i].__dict__:
                        Model_dict['arg' + str(count)] = node.args[i].__dict__['n']

                    if 'elts' in node.args[i].__dict__:
                        Model_dict['arg' + str(count)] = []
                        for j in node.args[i].__dict__['elts']:
                            if 'n' in j.__dict__:
                                Model_dict['arg' + str(count)] += [j.__dict__['n']]
                            if 's' in j.__dict__:
                                Model_dict['arg' + str(count)] += [j.__dict__['s']]
                for i in range(len(node.keywords)):
                    if 'arg' in node.keywords[i].__dict__:
                        if 'id' in node.keywords[i].__dict__['value'].__dict__:
                            if node.keywords[i].__dict__['value'].__dict__['id'] in Assign_dict:
                                Model_dict[node.keywords[i].__dict__['arg']] = \
                                    Assign_dict[node.keywords[i].__dict__['value'].__dict__['id']]
                        if 'n' in node.keywords[i].__dict__['value'].__dict__:
                            Model_dict[node.keywords[i].__dict__['arg']] = \
                                node.keywords[i].__dict__['value'].__dict__['n']
                        if 's' in node.keywords[i].__dict__['value'].__dict__:
                            Model_dict[node.keywords[i].__dict__['arg']] = \
                                node.keywords[i].__dict__['value'].__dict__['s']
                        if 'elts' in node.keywords[i].__dict__['value'].__dict__:
                            Model_dict[node.keywords[i].__dict__['arg']] = []
                            for j in node.keywords[i].__dict__['value'].__dict__['elts']:
                                if 'n' in j.__dict__:
                                    Model_dict[node.keywords[i].__dict__['arg']] += [j.__dict__['n']]
                                if 's' in j.__dict__:
                                    Model_dict[node.keywords[i].__dict__['arg']] += [j.__dict__['s']]
                                # CHU Y ID NUA
                        # (node.keywords[i].__dict__['arg'])

        if 'id' in node.func.__dict__:
            # print('zxczxcxzzcvzxvzx')
            if node.func.__dict__['id'] in self.apis:
                Model_dict['func'] = node.func.__dict__['id']
                for i in range(len(node.args)):
                    count += 1
                    if 'id' in node.args[i].__dict__:
                        if node.args[i].__dict__['id'] in Assign_dict:
                            Model_dict[node.args[i].__dict__['id']] = Assign_dict[node.args[i].__dict__['id']]
                            # print(node.args[i].__dict__['id'])
                    if 's' in node.args[i].__dict__:
                        Model_dict['arg' + str(count)] = node.args[i].__dict__['s']
                    if 'n' in node.args[i].__dict__:
                        Model_dict['arg' + str(count)] = node.args[i].__dict__['n']

                    if 'elts' in node.args[i].__dict__:
                        Model_dict['arg' + str(count)] = []
                        for j in node.args[i].__dict__['elts']:
                            if 'id' in j.__dict__:
                                if str(j.__dict__['id']) in Assign_dict:
                                    Model_dict['arg' + str(count)] += [Assign_dict[str(j.__dict__['id'])]]
                            if 'n' in j.__dict__:
                                Model_dict['arg' + str(count)] += [j.__dict__['n']]
                            if 's' in j.__dict__:
                                Model_dict['arg' + str(count)] += [j.__dict__['s']]
                for i in range(len(node.keywords)):
                    if 'arg' in node.keywords[i].__dict__:
                        if 'id' in node.keywords[i].__dict__['value'].__dict__:
                            if node.keywords[i].__dict__['value'].__dict__['id'] in Assign_dict:
                                Model_dict[node.keywords[i].__dict__['arg']] = \
                                    Assign_dict[node.keywords[i].__dict__['value'].__dict__['id']]
                        if 'n' in node.keywords[i].__dict__['value'].__dict__:
                            Model_dict[node.keywords[i].__dict__['arg']] = \
                                node.keywords[i].__dict__['value'].__dict__['n']
                        if 's' in node.keywords[i].__dict__['value'].__dict__:
                            Model_dict[node.keywords[i].__dict__['arg']] = \
                                node.keywords[i].__dict__['value'].__dict__['s']
                        if 'elts' in node.keywords[i].__dict__['value'].__dict__:
                            Model_dict[node.keywords[i].__dict__['arg']] = []
                            for j in node.keywords[i].__dict__['value'].__dict__['elts']:
                                #print(j.__dict__, 'aaaaaaaaaa')
                                if 'id' in j.__dict__:
                                    if str(j.__dict__['id']) in Assign_dict:
                                        Model_dict[node.keywords[i].__dict__['arg']] += [Assign_dict[str(j.__dict__['id'])]]
                                if 'n' in j.__dict__:
                                    Model_dict[node.keywords[i].__dict__['arg']] += [j.__dict__['n']]
                                if 's' in j.__dict__:
                                    Model_dict[node.keywords[i].__dict__['arg']] += [j.__dict__['s']]
                                # CHU Y ID NUA
                        # print(node.keywords[i].__dict__['arg'])
        if 'func' in node.func.__dict__:
            if 'id' in node.func.func.__dict__:
                if node.func.func.__dict__['id'] in self.apis:
                    Model_dict['func'] = node.func.func.__dict__['id']

                    for i in range(len(node.func.args)):
                        count += 1
                        if 'id' in node.func.args[i].__dict__:
                            if node.func.args[i].__dict__['id'] in Assign_dict:
                                Model_dict[node.func.args[i].__dict__['id']] = Assign_dict[
                                    node.func.args[i].__dict__['id']]
                                # print(node.func.args[i].__dict__['id'])
                        if 's' in node.func.args[i].__dict__:
                            Model_dict['arg' + str(count)] = node.func.args[i].__dict__['s']
                        if 'n' in node.func.args[i].__dict__:
                            Model_dict['arg' + str(count)] = node.func.args[i].__dict__['n']

                        if 'elts' in node.func.args[i].__dict__:
                            Model_dict['arg' + str(count)] = []
                            for j in node.func.args[i].__dict__['elts']:
                                if 'id' in j.__dict__:
                                    if str(j.__dict__['id']) in Assign_dict:
                                        Model_dict['arg' + str(count)] += [Assign_dict[str(j.__dict__['id'])]]
                                if 'n' in j.__dict__:
                                    Model_dict['arg' + str(count)] += [j.__dict__['n']]
                                if 's' in j.__dict__:
                                    Model_dict['arg' + str(count)] += [j.__dict__['s']]
                    for i in range(len(node.func.keywords)):
                        if 'arg' in node.func.keywords[i].__dict__:
                            if 'id' in node.func.keywords[i].__dict__['value'].__dict__:
                                if node.func.keywords[i].__dict__['value'].__dict__['id'] in Assign_dict:
                                    Model_dict[node.func.keywords[i].__dict__['arg']] = \
                                        Assign_dict[node.func.keywords[i].__dict__['value'].__dict__['id']]
                            if 'n' in node.func.keywords[i].__dict__['value'].__dict__:
                                Model_dict[node.func.keywords[i].__dict__['arg']] = \
                                    node.func.keywords[i].__dict__['value'].__dict__['n']
                            if 's' in node.func.keywords[i].__dict__['value'].__dict__:
                                Model_dict[node.func.keywords[i].__dict__['arg']] = \
                                    node.func.keywords[i].__dict__['value'].__dict__['s']
                            if 'elts' in node.func.keywords[i].__dict__['value'].__dict__:

                                Model_dict[node.func.keywords[i].__dict__['arg']] = []
                                for j in node.func.keywords[i].__dict__['value'].__dict__['elts']:
                                    if 'id' in j.__dict__:
                                        if str(j.__dict__['id']) in Assign_dict:
                                            Model_dict[node.func.keywords[i].__dict__['arg']] += [
                                                Assign_dict[str(j.__dict__['id'])]]
                                    if 'n' in j.__dict__:
                                        Model_dict[node.func.keywords[i].__dict__['arg']] += [j.__dict__['n']]
                                    if 's' in j.__dict__:
                                        Model_dict[node.func.keywords[i].__dict__['arg']] += [j.__dict__['s']]
                                    # CHU Y ID NUA
                            # print(node.func.keywords[i].__dict__['arg'])
        if self.saver != None:
            # print('zzzzzzzzzzzzzzzzzzzzzzzzz')
            f = open(self.saver, "a", encoding="ISO-8859-1")
            if len(Model_dict) > 0:

                # for i in Model_dict:
                if 'input_shape' in Model_dict:
                    # print('aaaaaaaa')
                    Model_dict.update({'Check': True})
                f.write(str(Model_dict) + '\n')
                '''
                if Model_dict[i] == '--weight_decay':
                    f.write(str(Model_dict) + '\n')
                if Model_dict[i] == '--lr':
                    f.write(str(Model_dict) + '\n')
                if Model_dict[i] == '--momentum':
                    f.write(str(Model_dict) + '\n')
                if Model_dict[i] in optimizer:
                    f.write(str(Model_dict) + '\n')
                    #f.write(str({'END':'END'}) + '\n')
                if Model_dict[i] in CNN:
                    f.write(str(Model_dict) + '\n')
                    #f.write(str({'END':'END'}) + '\n')
                '''
            f.close()
        #print(Model_dict, 'asdad')
        Model_dict.clear()
        block.next = next_block
        self.use_block(next_block)

    def do_If(self, node):
        ''' If an if statement is the last in a straight line then an empty
            and unused block will be created as the after_if. '''
        if_block = self.current_block
        after_if_block = self.new_block()
        # Then block
        then_block = self.new_block()
        self.add_to_exits(if_block, then_block)
        self.use_block(then_block)
        for z in node.body:
            self.visit(z)
        # Make sure the then exits point to the correct place
        self.check_child_exits(self.current_block, after_if_block)
        # Else block
        if node.orelse:
            else_block = self.new_block()
            self.add_to_exits(if_block, else_block)
            self.use_block(else_block)
            for z in node.orelse:
                self.visit(z)
            # Make sure the else exits point to the correct place
            self.check_child_exits(self.current_block, after_if_block)
        else:
            self.add_to_exits(if_block, after_if_block)
        # Set the next block of the if to the after_if block
        if_block.next = after_if_block
        self.use_block(after_if_block)

    def do_While(self, node):
        self.do_Loop(node)

    def do_For(self, node):
        self.do_Loop(node)

    def do_Loop(self, node):
        iter_array = []
        # print(node.__dict__, 'Loop')
        test_block = self.current_block

        test_block.tag = Block.LOOP_HEADER
        self.push_frame_block(F_BLOCK_LOOP, test_block)

        after_loop_block = self.new_block()
        loop_body_block = self.new_block()
        self.add_to_exits(test_block, loop_body_block)
        test_block.next = after_loop_block
        self.use_block(loop_body_block)

        # self.visit(node.target)
        # self.visit(node.iter)
        # print(node.iter.__dict__['args'])
        # for i in range(len(node.iter.__dict__['args'])):
        # iter_array.append(node.iter.__dict__['args'][i].__dict__['n'])
        # node.__dict__['target'] = node.target.__dict__['id']
        # node.__dict__['iter'] = iter_array
        # print(node.__dict__, 'Loop')
        for z in node.body:
            self.visit(z)

        self.check_child_exits(self.current_block, test_block)
        self.pop_frame_block(F_BLOCK_LOOP, test_block)

        if node.orelse:
            else_body = self.new_block()
            self.add_to_exits(test_block, else_body)
            self.use_block(else_body)
            else_body.next = after_loop_block
            for z in node.orelse:
                self.visit(z)
            self.check_child_exits(self.current_block, after_loop_block)
        else:
            self.add_to_exits(test_block, after_loop_block)

        self.use_next_block(after_loop_block)

    def do_AugAssign(self, node):
        block = self.current_block
        next_block = self.new_block()
        self.add_to_exits(block, next_block)
        block.next = next_block
        self.use_block(next_block)

    def do_Return(self, node):
        ''' End the current block here.
            No statements in this block after this are valid.
            In a try, returns go to the finally block. '''
        if node.value:
            self.visit(node.value)
        # Check if the block is an try-finally.
        for f_block_type, f_block in reversed(self.frame_blocks):
            if f_block_type == F_BLOCK_FINALLY:
                return_exit = f_block
                break
        else:
            return_exit = self.exit_block
        self.current_block.exit_blocks.append(return_exit)
        self.current_block.has_return = True

    def do_Continue(self, node):
        ''' Continues can not be in a finally block.
            TODO: Fix this up.  '''
        if not self.frame_blocks:
            self.error("'continue' not properly in loop", node)
        current_block, block = self.frame_blocks[-1]
        if current_block == F_BLOCK_LOOP:
            self.current_block.exit_blocks.append(block)
        elif current_block == F_BLOCK_EXCEPT or \
                current_block == F_BLOCK_FINALLY:
            # Find the loop
            for i in range(len(self.frame_blocks) - 2, -1, -1):
                f_type, block = self.frame_blocks[i]
                if f_type == F_BLOCK_LOOP:
                    self.current_block.exit_blocks.append(block)
                    break
                if f_type == F_BLOCK_FINALLY_END:
                    self.error("'continue' not supported inside 'finally' "
                               "clause", node)
            else:
                self.error("'continue' not properly in loop", node)
                return
        elif current_block == F_BLOCK_FINALLY_END:
            self.error("'continue' not supported inside 'finally' clause", node)
        self.current_block.has_return = True

    def do_Break(self, node):
        ''' A break can only be in a loop.
            A break causes the current block to exit to block after the loop
            header (its next) '''
        # Find first loop in stack
        for f_block_type, f_block in reversed(self.frame_blocks):
            if f_block_type == F_BLOCK_LOOP:
                self.current_block.exit_blocks.append(f_block.next)
                break
        else:
            self.error("'break' outside loop", node)
        self.current_block.has_return = True

    def do_Yield(self, node):
        ''' Here we deal with the control flow when the iterator goes through
            the function.
            We don't set has_return to true since, in theory, it can either
            exit or continue from here. '''
        self.current_block.exit_blocks.append(self.exit_block)
        next_block = self.new_block()
        self.current_block.exit_blocks.append(next_block)
        self.use_next_block(next_block)

    def do_Try(self, node):
        ''' It is a great ordeal to find out which statements can cause which
            exceptions. Assume every statement can cause any exception. So
            each statement has its own block and a link to each exception.

            orelse executed if an exception is not raised therefore last try
            statement should point to the else.

            nested try-finallys go to each other during a return
            TODO'''
        after_try_block = self.new_block()
        final_block = None
        try_body_block = self.new_block()
        self.current_block.next_block = try_body_block
        orelse_block = self.new_block()

        before_line_no = self.current_line_num
        if node.finalbody:
            # Either end of orelse or try should point to finally body
            final_block = self.new_block()
            self.use_block(final_block)
            self.push_frame_block(F_BLOCK_FINALLY_END, node)
            for z in node.finalbody:
                self.visit(z)
            self.pop_frame_block(F_BLOCK_FINALLY_END, node)
            self.check_child_exits(self.current_block, after_try_block)
        self.current_line_num = before_line_no

        before_line_no = self.current_line_num
        exception_handlers = []
        for handler in node.handlers:
            assert isinstance(handler, ast.ExceptHandler)
            initial_handler_block = self.new_block()
            self.use_block(initial_handler_block)
            for z in handler.body:
                self.visit(z)
            handler_exit = final_block if node.finalbody else after_try_block
            self.check_child_exits(self.current_block, handler_exit)
            exception_handlers.append(initial_handler_block)
        self.current_line_num = before_line_no

        f_blocks = []
        if node.finalbody:
            f_blocks.append((F_BLOCK_FINALLY, final_block))
        if node.handlers:
            f_blocks.append((F_BLOCK_EXCEPT, exception_handlers))
        for f in f_blocks:
            self.push_frame_block(f[0], f[1])
        self.use_block(try_body_block)
        for z in node.body:
            self.visit(z)
        for f in reversed(f_blocks):
            self.pop_frame_block(f[0], f[1])

        if node.orelse:
            orelse_block = self.new_block()
            # Last block in body can always go to the orelse
            self.check_child_exits(self.current_block, orelse_block)
            self.use_block(orelse_block)
            for z in node.orelse:
                self.visit(z)
            orelse_exit = final_block if node.finalbody else after_try_block
            self.check_child_exits(self.current_block, orelse_exit)
        else:
            self.check_child_exits(self.current_block, after_try_block)

        self.use_next_block(after_try_block)


if __name__ == '__main__':

    count = 0
    # repo_download()
    # mm = ModelMining()
    # mm.file_filter()
    # pyfiles = open("database_creation/py_with_model", encoding="ISO-8859-1")
    # for file_link in pyfiles:
    # count += 1
    # if count >= 0:
    cfg = ControlFlowGraph(Constant.kerasapis, 'opt.txt')

    try:
        s_ast = cfg.parse_file('test.py')
    except (TabError, SyntaxError):
        pass
    layer_count += 1
# collect_model()
