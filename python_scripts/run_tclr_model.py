from mpi4py import MPI
from tclr_model_spatial_val import parse_args, pipeline

# arglist = [["all", "-d", str(i), "--assim_freq", "weekly"] for i in range(0,11)]
# arglist.extend([["all", "-d", str(i), "--assim_freq", "seasonally"] for i in range(0,11)])
# arglist.extend([["all", "-d", str(i), "--assim_freq", "monthly"] for i in range(0,11)])
# arglist.extend([["all", "-d", str(i), "--assim_freq", "daily"] for i in range(0,11)])
# arglist.extend([["all", "-d", str(i)] for i in range(0,11)])
   
# arglist = [["all", "-d", "3", "--assim_freq", i] for i in ["weekly", "monthly", "seasonally", "daily"]]
# arglist.append(["all", "-d", "3"])

arglist = [["all", "-d", "3", "--train_prop", i] for i in [0.25, 0.33333, 0.5, 0.66667, 0.75]]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

njobs = len(arglist)

my_jobs = list(range(rank, njobs, nprocs))

for job in my_jobs:
    args = parse_args(arglist[job])
    pipeline(args)
