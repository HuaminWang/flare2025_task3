import os
import sys
import atexit
import functools
import json
import traceback
import time
import inspect

import torch

if os.environ.get("IMPORT_TOOL", None) is None:
    os.environ['IMPORT_TOOL'] = "1"
    all_op = {}
    user_op = {}
    depth = 0
    START_TIME = time.time()
    MAX_TIME = 10 * 60
    max_err_stdout_line = int(os.environ.get("MAX_ERR_STDOUT_LINE", 15))
    screen_log = ''


    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout

        def write(self, message):
            self.terminal.write(message)
            global screen_log
            screen_log += message

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            # you might want to specify some extra behavior here.
            pass


    sys.stdout = Logger()


    def check_time():
        if time.time() - START_TIME > MAX_TIME:
            os.environ['FINISH'] = "1"
            exit()


    def _add_op(d, prefix, op_name):
        if prefix not in d:
            # use list because set is not JSON serializable
            d[prefix] = [op_name]
        elif op_name not in d[prefix]:
            d[prefix].append(op_name)


    def add_op(prefix, op_name, record=True):
        _add_op(all_op, prefix, op_name)
        global depth
        if record and depth == 0:
            _add_op(user_op, prefix, op_name)


    def func_warpper(func, prefix=None, name=None, record=True, step_warpper=False, call=False,
                     is_instance_methord=False):
        # global depth
        # if depth != 0:
        #     return func
        if call and (hasattr(func, '__getitem__') or hasattr(func, '__getattr__')):
            # @functools.wraps(func)
            class WP_FUNC():
                def __init__(self, f, prefix, name, record):
                    self.f = f
                    self.prefix = prefix
                    self.name = name
                    self.record = record

                def __call__(self, *args, **kwargs):
                    check_time()
                    add_op(self.prefix, self.name, self.record)
                    global depth
                    # if self.record and depth == 0:
                    #     if self.prefix not in all_op:
                    #         all_op[self.prefix] = []
                    #     if self.name not in all_op[self.prefix]:
                    #         all_op[self.prefix].append(self.name)
                    depth = depth + 1
                    res = self.f(*args, **kwargs)
                    depth = depth - 1
                    return res

                def __getitem__(self, ind):
                    return self.f.__getitem__(ind)

                def __setitem__(self, ind, data):
                    return self.f.__setitem__(ind, data)
                # def __getattr__(self, attr):
                #     return self.f.__attr__(attr)
                # def __setattr__(self, attr, data):
                #     return self.f.__setattr__(attr, data)

            return WP_FUNC(func, prefix, name, record)
        else:
            @functools.wraps(func)
            def wp_func(*args, **kwargs):
                check_time()
                if step_warpper:
                    if hasattr(args[0], '_my_iter'):
                        it = getattr(args[0], '_my_iter')
                        if it == 3:
                            os.environ['FINISH'] = "1"
                            exit()
                        setattr(args[0], '_my_iter', it + 1)
                    else:
                        setattr(args[0], '_my_iter', 1)
                add_op(prefix, name, record)
                global depth
                # if record and depth == 0:
                #     if prefix not in all_op:
                #         all_op[prefix] = []
                #     if name not in all_op[prefix]:
                #         all_op[prefix].append(name)
                depth = depth + 1
                if is_instance_methord:
                    res = func(*args[1:], **kwargs)
                else:
                    res = func(*args, **kwargs)
                depth = depth - 1
                return res

            return wp_func


    class Wrapper(type(torch)):
        def __init__(self, wrapped, name=None):
            if name is None:
                name = wrapped.__name__
            self.name = name
            self.wrapped = wrapped
            self.visited = {}

        def __dir__(self):
            return dir(self.wrapped)

        def __getattr__(self, name):
            check_time()
            prefix = self.name + '.' + name
            if name in self.visited:
                return self.visited[name]
            else:

                if prefix in sys.modules and prefix != 'torch.tensor':
                    temp = sys.modules[prefix]
                else:
                    temp = getattr(self.wrapped, name)

                # temp = getattr(self.wrapped, name)

                if name.startswith("_"):
                    self.visited[name] = temp
                    return temp

                # global depth
                # if depth == 0:
                #     if self.name not in all_op:
                #         all_op[self.name] = []
                #     all_op[self.name].append(name)
                add_op(self.name, name)

                # something need fix, maybe about some @staticmethod of torch.autograd.Function
                # if prefix == 'torch.autograd':
                #     self.visited[name] = temp
                #     return temp

                # bug need fixed
                # if prefix.startswith('torch.jit.annotations'):
                #     self.visited[name] = temp
                #     return temp

                # bug need fixed when import torchvision
                # source /mnt/lustre/share/spring/r0.3.2
                if prefix.startswith('torch.jit.script'):
                    @functools.wraps(temp)
                    def a(func):
                        add_op('torch.jit', 'script')
                        return func

                    self.visited[name] = a
                    return a

                # bug need fixed
                # if name == 'default_collate':
                if prefix.startswith('torch.utils.data._utils'):
                    self.visited[name] = temp
                    return temp

                # for torch.cuda.set_device
                # if name == 'set_device':
                #     self.visited[name] = temp
                #     return temp

                if isinstance(temp, type(sys)):
                    # wp = Wrapper(temp, prefix)
                    # self.visited[name] = wp
                    # return wp
                    if prefix == 'torch.cuda':
                        temp = Wrapper(temp, prefix)
                    self.visited[name] = temp
                    return temp
                elif isinstance(temp, type):
                    for m in dir(temp):
                        if not m.startswith("_"):
                            try:
                                func = getattr(temp, m)
                                if callable(func):
                                    setattr(temp, m,
                                            func_warpper(func, prefix, m,
                                                         step_warpper=m == 'step' and issubclass(temp,
                                                                                                 torch.optim.Optimizer),
                                                         is_instance_methord=isinstance(inspect.getattr_static(temp, m),
                                                                                        staticmethod)))
                            except:
                                pass
                    try:
                        class MC(type(temp)):
                            def __instancecheck__(self, a):
                                return isinstance(a, temp)

                        class WP(temp, meatclass=MC):
                            _real_class = temp

                            # __module__ = temp.__module__
                            # __class__ = temp.__class__
                            def __getattr__(self, attr):
                                check_time()

                                add_op(prefix, attr)
                                # global depth
                                # if depth == 0 and attr not in all_op[prefix]:
                                #     all_op[prefix].append(attr)

                                try:
                                    return super().__getattr__(attr)
                                except:
                                    try:
                                        return self.__dict__[attr]
                                    except:
                                        raise AttributeError

                        self.visited[name] = WP
                        return WP
                    except:
                        self.visited[name] = temp
                        return temp
                else:
                    if callable(temp):
                        temp = func_warpper(temp, self.name, name, False, call=True)
                    self.visited[name] = temp
                    return temp


    class ExitHooks(object):
        def __init__(self):
            self.exc_type = None
            self.exc_value = None
            self.exc_traceback = None

        def hook(self):
            sys.excepthook = self.exc_handler

        def exc_handler(self, exc_type, exc_value, exc_traceback):
            self.exc_type = exc_type
            self.exc_value = exc_value
            self.exc_traceback = exc_traceback
            sys.__excepthook__(exc_type, exc_value, exc_traceback)


    exit_hooks = ExitHooks()
    exit_hooks.hook()


    def exit_print(exit_hook):
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0
        if rank != 0:
            return
        finish = os.environ.get('FINISH', False)
        if finish:
            report = "op analysing finished"
            op_str = json.dumps(all_op)
            user_op_str = json.dumps(user_op)
            # print("final all_op:", all_op)
        else:
            report = "op analysing failed"
        print(report + "!!!")

        file_name = os.environ.get("OP_ANALYSING_FILE", "all_ops")
        label = os.environ.get("MODEL_LABEL", "new_model:")
        with open(file_name, "a") as f:
            f.write(label + '\n')
            f.write(report + '\n')
            f.write(str(sys.argv) + '\n')
            if finish:
                f.write("all_ops:")
                f.write(op_str + '\n')
                f.write("user_ops:")
                f.write(user_op_str + '\n')
            else:
                line_log = screen_log.split('\n')[:-2]
                if len(line_log) > max_err_stdout_line:
                    log = str(max_err_stdout_line) + " line"
                    if max_err_stdout_line > 1:
                        log += "s"
                    log += " of stdout are showed below, other "
                    d = len(line_log) - max_err_stdout_line
                    log += str(d) + " line"
                    if d > 1:
                        log += "s"
                    f.write("-" * 10 + " only last " + log + " is omitted " + "-" * 10 + '\n')
                else:
                    f.write("-" * 10 + " stdout " + "-" * 10 + '\n')
                for l in line_log[-max_err_stdout_line:]:
                    f.write(l + '\n')
                f.write("-" * 10 + " python excption " + "-" * 10 + '\n')
                traceback.print_exception(exit_hook.exc_type, exit_hook.exc_value,
                                          exit_hook.exc_traceback, file=f)
            f.write('\n')


    for p in sys.modules.keys():
        if p == 'torch' or (p.startswith('torch.')
                            and not p.startswith('torch.autograd')
                            and not p.startswith('torch.cuda')
                            # and not p.startswith('torch._C')
                            and not p.startswith('torch.multiprocessing')
        ):
            sys.modules[p] = Wrapper(sys.modules[p])

    atexit.register(exit_print, exit_hooks)