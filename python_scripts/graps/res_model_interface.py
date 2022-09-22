from __future__ import print_function

from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import ctypes as ct
import os
import sys
import pprint
from collections import defaultdict
from IPython import embed as II
pp = pprint.PrettyPrinter()


class ReservoirModel(object):
    """
    ReservoirModel object provides all the required functions to create the link to the
    fortran libray containing the reservoir model, initialize the model instance, and
    simulate as many times as the user desires. This file must be executed within the
    folder containing the compiled reservoir model library and all the requried input
    files.
    """

    def __init__(self, n_init_params, input_path="./", output_path="./", func_flag="mhb"):
        self.n_init_params = n_init_params
        self.in_path = input_path
        self.out_path = output_path
        self.UpdatePathFile
        self.library = ct.CDLL(
            '/home/lcford2/temoaMultiRes/TVA_Res_Model/res_model.so')
        self.py_init_params = [0.0 for i in range(n_init_params)]
        self.ArrayType1 = ct.c_double * n_init_params
        self.initial_params = (self.ArrayType1)(*self.py_init_params)
        self.py_hydro_benefit = [1.0 for i in range(int(n_init_params))]
        self.init_index_cons_ct = ct.c_int(1)
        self.n_init_params_ct = ct.c_int(self.n_init_params)
        self.nres_ct = ct.c_int(0)
        self.nuser_ct = ct.c_int(0)
        self.nrestr_ct = ct.c_int(0)
        self.init_fun = self.library.initialize_
        self.simul_fun = self.library.python_simulate_
        self.opt_fun = self.library.python_optimize_
        self.func_flag_map = {
            "mhb": 1,    # max hydro benefits
            "mhp": 2,    # max hydropower
            "msd": 3,    # min spill and deficit
            "mrl": 4     # max release
        }
        self.func_flag_str = func_flag
        self.output_dict = dict()
        self.new_max_act = dict()
        self.res_names = dict()
        self.temoa_names = dict()

    @property
    def UpdatePathFile(self):
        with open("./path.dat", "w") as f:
            f.write(self.in_path + "\n")
            f.write(self.out_path + "\n")
        return

    @property
    def DeclareInitArgumentTypes(self):
        """
        Declaring argument types and return types for init_fun.
        The only argument is an array of double precision decimals
        and there is no return value.
        """
        self.init_fun.argtypes = [ct.POINTER(ct.c_int),
                                  ct.POINTER(ct.c_int),
                                  self.ArrayType1,
                                  ct.POINTER(ct.c_int),
                                  ct.POINTER(ct.c_int),
                                  ct.POINTER(ct.c_int)]
        self.init_fun.restype = None
        return None

    @property
    def Initialize(self):
        _ = self.init_fun(self.n_init_params_ct,
                          self.init_index_cons_ct,
                          self.initial_params,
                          self.nres_ct,
                          self.nuser_ct,
                          self.nrestr_ct)
        self.nparam, self.index_cons = self.n_init_params_ct.value, self.init_index_cons_ct.value
        self.nres, self.nuser, self.nrestr = (self.nres_ct.value,
                                              self.nuser_ct.value,
                                              self.nrestr_ct.value)
        self.dec_vars = self.initial_params
        return None

    @property
    def GetResNames(self):
        with open(os.path.join(self.out_path, 'id_name.out'), 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                id_num, name = line.split(',')
                self.res_names[int(id_num)] = name.strip()
        return None

    @property
    def GetTemoaNames(self):
        temoa_names = {
            'Apalachia H': 'Apalachia_HY_TN',
            'BlueRidge H': 'BlueRidge_HY_GA',
            'Boone H': 'Boone_HY_TN',
            'Chatuge H': 'Chatuge_HY_NC',
            'Cherokee H': 'Cherokee_HY_TN',
            'Chikamauga H': 'Chickamauga_HY_TN',
            'Douglas H': 'Douglas_HY_TN',
            'Fontana H': 'Fontana_HY_NC',
            'FtLoudoun H': 'FortLoudoun_HY_TN',
            'FtPatrick H': 'FortPatrick_HY_TN',
            'Guntersville H': 'Guntersville_HY_AL',
            'Hiwassee H': 'Hiwassee_HY_NC',
            'Kentucky H': 'Kentucky_HY_KY',
            'MeltonH H': 'MeltonHill_HY_TN',
            'Nikajack H': 'Nickajack_HY_TN',
            'Norris H': 'Norris_HY_TN',
            'Nottely H': 'Nottely_HY_GA',
            'Ocoee1 H': 'Ocoee1_HY_TN',
            'Ocoee3 H': 'Ocoee3_HY_TN',
            'Pickwick H': 'PickwickLanding_HY_TN',
            'RacoonMt H': 'RaccoonMt_Storage_TN',
            'SHolston H': 'SouthHolston_HY_TN',
            'TimsFord H': 'TimsFord_HY_TN',
            'WattsBar H': 'WattsBar_HY_TN',
            'Watuga H': 'Watauga_HY_TN',
            'Wheeler H': 'Wheeler_HY_AL',
            'Wilbur H': 'Wilbur_HY_TN',
            'Wilson H': 'Wilson_HY_AL'
        }
        for key in list(self.res_names.keys()):
            self.temoa_names[key] = temoa_names[self.res_names[key]]
        return None

    def CreateNewMaxAct(self, ntime=3):
        month_key = [2011 + i for i in range(ntime)]
        for key, name in self.temoa_names.items():
            values = self.last_results[name]
            for m_index, value in enumerate(values):
                try:
                    index = (month_key[m_index], name)
                    self.new_max_act[index] = value
                except IndexError as e:
                    print('bad', m_index)
        return None

    @property
    def CreateSimulArguments(self):
        from IPython import embed as II
        self.dec_vars_ct = self.ArrayType2(*self.dec_vars)
        self.nparam_ct = ct.c_int(int(self.nparam))
        self.index_cons_ct = ct.c_int(int(self.index_cons))
        self.gcons_ct = ct.c_double(0.0)
        self.hydro_benefit = self.ArrayType2(*self.py_hydro_benefit)
        self.id_output = self.ArrayType2_int(
            *[0 for i in range(int(self.nparam))])
        self.value_output = self.ArrayType2(
            *[0.0 for i in range(int(self.nparam))])
        self.ncons = self.nres + self.nuser + self.nrestr
        self.cons_values = self.ArrayType3(
            *[0.0 for i in range(int(self.ncons))])
        self.cons_id = self.ArrayType3_int(
            *[0 for i in range(int(self.ncons) - self.nrestr)])
        self.cons_mag = self.ArrayType4(
            *[0 for i in range(self.nres)])
        self.min_rel = self.ArrayType5(
            *[0.0 for i in range(self.nuser)])
        self.max_rel = self.ArrayType5(
            *[0.0 for i in range(self.nuser)])
        self.user_id = self.ArrayType5_int(
            *[0 for i in range(self.nuser)])
        self.spill_values = self.ArrayType2(
            *[0.0 for i in range(self.nparam)])
        self.deficit_values = self.ArrayType2(
            *[0.0 for i in range(self.nparam)])
        self.res_id_for_spdef = self.ArrayType2_int(
            *[0 for i in range(self.nparam)])
        self.func_flag = ct.c_int(self.func_flag_map[self.func_flag_str])
        return None

    @property
    def DeclareSimulArgumentTypes(self):
        self.ArrayType2 = ct.c_double * int(self.nparam)
        self.ArrayType2_int = ct.c_int * int(self.nparam)
        self.ArrayType3 = ct.c_double * \
            int(self.nres + self.nuser + self.nrestr)
        self.ArrayType3_int = ct.c_int * \
            int(self.nres + self.nuser)
        self.ArrayType4 = ct.c_double * int(self.nres)
        self.ArrayType5 = ct.c_double * int(self.nuser)
        self.ArrayType5_int = ct.c_int * int(self.nuser)
        self.simul_fun.argtypes = [ct.POINTER(ct.c_int),
                                   ct.POINTER(ct.c_int),
                                   ct.POINTER(self.ArrayType2),
                                   ct.POINTER(ct.c_double),
                                   ct.POINTER(self.ArrayType2),
                                   ct.POINTER(self.ArrayType2_int),
                                   ct.POINTER(self.ArrayType2),
                                   ct.POINTER(self.ArrayType3),
                                   ct.POINTER(self.ArrayType3_int),
                                   ct.POINTER(self.ArrayType4),
                                   ct.POINTER(self.ArrayType5),
                                   ct.POINTER(self.ArrayType5),
                                   ct.POINTER(self.ArrayType5_int),
                                   ct.POINTER(self.ArrayType2),
                                   ct.POINTER(self.ArrayType2),
                                   ct.POINTER(self.ArrayType2_int),
                                   ct.POINTER(ct.c_int)]
        self.opt_fun.argtypes = self.simul_fun.argtypes
        self.simul_fun.restype = None
        self.opt_fun.restype = None
        return None

    @property
    def Simulate(self):
        _ = self.simul_fun(self.nparam_ct,
                           self.index_cons_ct,
                           self.dec_vars_ct,
                           self.gcons_ct,
                           self.hydro_benefit,
                           self.id_output,
                           self.value_output,
                           self.cons_values,
                           self.cons_id,
                           self.cons_mag,
                           self.min_rel,
                           self.max_rel,
                           self.user_id,
                           self.spill_values,
                           self.deficit_values,
                           self.res_id_for_spdef,
                           self.func_flag)

    @property
    def Optimize(self):
        _ = self.opt_fun(self.nparam_ct,
                         self.index_cons_ct,
                         self.dec_vars_ct,
                         self.gcons_ct,
                         self.hydro_benefit,
                         self.id_output,
                         self.value_output,
                         self.cons_values,
                         self.cons_id,
                         self.cons_mag,
                         self.min_rel,
                         self.max_rel,
                         self.user_id,
                         self.spill_values,
                         self.deficit_values,
                         self.res_id_for_spdef,
                         self.func_flag)

    def MakeOutputContainers(self, run_name):
        self.output_dict[run_name] = dict()
        self.last_results = dict()
        self.res_cons = [self.cons_values[i] for i in range(self.nres)]
        self.user_cons = [self.cons_values[i]
                          for i in range(self.nres, self.nuser + self.nres)]
        self.restr_cons = [self.cons_values[i]
                           for i in range(self.nuser + self.nres, self.ncons)]
        self.res_con_id = [self.cons_id[i] for i in range(self.nres)]
        self.user_con_id = [self.cons_id[i] for i in range(
            self.nres, self.nuser + self.nres)]
        self.res_viols = [self.cons_mag[i] for i in range(self.nres)]
        self.min_rel = [i for i in self.min_rel]
        self.max_rel = [i for i in self.max_rel]
        self.user_id = [i for i in self.user_id]
        self.spill = [i for i in self.spill_values]
        self.deficit = [i for i in self.deficit_values]
        self.spdef_ids = [i for i in self.res_id_for_spdef]
        self.release_bounds = dict()
        self.res_violations = dict()
        self.res_constraints = dict()
        self.user_constraints = dict()
        self.spill_dict = defaultdict(list)
        self.deficit_dict = defaultdict(list)
        block_ids = self.id_output[:]
        values = self.value_output[:]
        for b_id, value in zip(block_ids, values):
            name = self.temoa_names[b_id]
            if self.output_dict[run_name].get(name) == None:
                self.output_dict[run_name][name] = [value]
            else:
                self.output_dict[run_name][name].append(value)
            if self.last_results.get(name) == None:
                self.last_results[name] = [value]
            else:
                self.last_results[name].append(value)
        for res_id, res_con in zip(self.res_con_id, self.res_cons):
            self.res_constraints[res_id] = res_con
        for user_id, user_con in zip(self.user_con_id, self.user_cons):
            self.user_constraints[user_id] = user_con
        for res_id, res_viol in zip(self.res_con_id, self.res_viols):
            self.res_violations[res_id] = res_viol
        for u_id, min_rel, max_rel in zip(self.user_id, self.min_rel, self.max_rel):
            self.release_bounds[u_id] = (min_rel, max_rel)
        for res_id, spill, deficit in zip(self.spdef_ids, self.spill, self.deficit):
            self.spill_dict[res_id].append(spill)
            self.deficit_dict[res_id].append(deficit)
        # II()
        return None

    def PPrintOutput(self, run_name=None):
        if run_name == None:
            pp.pprint(self.output_dict)
        else:
            pp.pprint(self.output_dict[run_name])

    def PrintOutput(self, run_name=None):
        def out_format(run, res, values):
            list_format = "{:<10.2f} " * \
                (len(values)) + "{units:<s}"  # -1)+"{:.2f} MWh"
            output_format = " " * 6 + "{:8s} {:24s} " + list_format
            return output_format.format(run, res, *values, units='MWh/month')
        if run_name == None:
            for run in sorted(self.output_dict.keys()):
                print(" " + "-" * 38, run.upper(), "-" * 38)
                for res, values in list(self.output_dict[run].items()):
                    print(out_format(run, res, values))
        else:
            print(" " + "-" * 38, run_name.upper(), "-" * 38)
            for res, values in list(self.output_dict[run_name].items()):
                print(out_format(run_name, res, values))

    def __repr__(self):
        return "ReservoirModel({},{},{})".format(self.n_init_params, self.in_path, self.out_path)


def InitializeModel(res_model):
    res_model.DeclareInitArgumentTypes
    res_model.Initialize
    res_model.GetResNames
    res_model.GetTemoaNames
    res_model.DeclareSimulArgumentTypes
    return None


def Simulate(res_model, run_name):
    res_model.CreateSimulArguments
    res_model.Simulate
    res_model.MakeOutputContainers(run_name)
    return None


def Optimize(res_model, run_name):
    res_model.CreateSimulArguments
    res_model.Optimize
    res_model.MakeOutputContainers(run_name)
    return None


def main(n_init_params, num_runs, input_path, output_path, run_names=None, run_flag="simul"):
    res_model = ReservoirModel(n_init_params, input_path, output_path)
    InitializeModel(res_model)
    if run_flag == "simul":
        run_function = Simulate
    elif run_flag == "opt":
        run_function = Optimize
    else:
        sys.exit(
            "The provided run flag is not supported. Please provide either 'simul' or 'opt'")
    if run_names == None:
        run_names = ['run_' + str(i) for i in range(1, num_runs + 1)]
    if len(run_names) != num_runs:
        sys.exit(
            'The number of names provided does not equal the number of runs specified. Please revise.')
    else:
        for run in range(num_runs):
            run_function(res_model, run_names[run])
        res_model.PrintOutput()
    return res_model.output_dict


def time(n_init_params, iterations, input_path, output_path):
    """The function is implemented solely to compare the times between
    running the initialization step and then simulating versus initializing
    and simulating for every call.

    Arguments:
        n_init_params {int} -- number of parameters for res_model
        iterations {int} -- how many times to run each model

    Returns:
        dict -- contains times for both models (new/old)
    """
    from time import time as timer

    def time_new():
        res_model = ReservoirModel(n_init_params, input_path, output_path)
        time = 0
        start = timer()
        InitializeModel(res_model)
        for i in range(iterations):
            Simulate(res_model, i)
        end = timer()
        time = end - start
        return time

    def time_old():
        time = 0
        start = timer()
        for i in range(iterations):
            os.system('./test')
        end = timer()
        time = end - start
        return time
    return {'old_time': time_old(), 'new_time': time_new()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        "Python module for running the Fortran multireservoir model GRAPS")
    parser.add_argument("nparam", help="Number of parameters for the model. (nuser * ntime)",
                        action="store", type=int)
    parser.add_argument("--nruns", "-NR", help="Number of runs to perform. Default = 1",
                        action="store", type=int, default=1)
    parser.add_argument("--scenario", "-S", help="Modeled scenario for input and output data.",
                        action="store", type=str)
    parser.add_argument("--runflag", "-R", help="Run flag for optimization or simulation. Default = 'simul'",
                        action="store", type=str, choices=["simul", "opt"], default="simul")
    parser.add_argument("--time", "-T", help="Flag for timing the model",
                        action="store_true")
    parser.add_argument("--optfun", "-O", type=str, choices=["mhb", "mhp", "msd", "mrl"], default="mhp",
                        help="Choose optimization function for FSQP")

    args = parser.parse_args()

    in_path = os.path.join(".", "ReservoirInput", args.scenario) + "/"
    out_path = os.path.join(".", "ReservoirOutput", args.scenario) + "/"

    # II()

    if args.time:
        iterations = input(
            "Please enter the number of iterations for the timer: ")
        times = time(args.nparam, int(iterations), in_path, out_path)
        pp.pprint(times)

    else:
        main(args.nparam, args.nruns, in_path, out_path, run_flag=args.runflag)
