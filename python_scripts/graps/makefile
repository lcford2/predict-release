
all: res_model multireservoir

res_model:
	ifort -check all -traceback -shared -fpic path_vars.f90 definitions.f90 My_variables.f90 ffsqp.f qld.f init.f90 all_simul.f90 -ores_model.so

multireservoir:
	ifort -check all -traceback path_vars.f90 definitions.f90 My_variables.f90 ffsqp.f qld.f init.f90 all_simul.f90 multireservoir.f90 -o multireservoir

clean:
	rm -f *.o standard_* *.out *.mod res_model.so multireservoir
