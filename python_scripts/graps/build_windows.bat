@ECHO OFF

cd src
mkdir build
xcopy *.f* build
cd build
ifort /check:all /traceback  type_Node.f90 type_Stack.f90 calpak.f90 ffsqp.f qld.f^
 path_vars.f90 definitions.f90 My_variables.f90 all_simul.f90 init.f90 multireservoir.f90 /exe:multireservoir

del *.f*
move * ..\..\bin
cd ..\..
rmdir src\build
