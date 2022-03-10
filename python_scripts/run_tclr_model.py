from mpi4py import MPI
from tclr_model import parse_args, pipeline

# arglist = [["all", "-d", str(i), "--assim_freq", "seasonally"] for i in range(1,11)]
arglist = [["all", "-d", "0", "--assim_freq", i] for i in ["weekly", "monthly", "seasonally"]]
arglist.append(["all", "-d", "0"])


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

njobs = len(arglist)

my_jobs = list(range(rank, njobs, nprocs))

for job in my_jobs:
    args = parse_args(arglist[job])
    pipeline(args)
