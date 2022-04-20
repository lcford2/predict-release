from mpi4py import MPI
from tclr_model import parse_args, pipeline

# arglist = [["all", "-d", str(i), "--assim_freq", "weekly"] for i in range(0,11)]
# arglist.extend([["all", "-d", str(i), "--assim_freq", "seasonally"] for i in range(0,11)])
# arglist.extend([["all", "-d", str(i), "--assim_freq", "monthly"] for i in range(0,11)])
# arglist.extend([["all", "-d", str(i), "--assim_freq", "daily"] for i in range(0,11)])
arglist = [["all", "-d", str(i)] for i in range(1,11)]
   
# arglist = [["all", "-d", "4", "--assim", i] for i in ["daily", "weekly", "monthly", "seasonally", "semi-annually", "yearly"]]
# arglist.append(["all", "-d", "4"])

# train_props = [i / 100 for i in range(25, 91, 5)]
# arglist = [["all", "-d", "3", "--train_prop", str(i)] for i in train_props]

#arglist = [["all", "-d", "3", "--train_prop", "0.75", "--seed", str(i)] for i in range(1000)]
#arglist = arglist[:1]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

njobs = len(arglist)

my_jobs = list(range(rank, njobs, nprocs))

for job in my_jobs:
    args = parse_args(arglist[job])
    pipeline(args)
