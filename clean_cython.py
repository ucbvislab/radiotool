import glob
from subprocess import check_output

try:
    check_output("rm -f radiotool/algorithms/build_table_mem_efficient.c", shell=True)
    check_output("rm -f radiotool/algorithms/build_table_mem_efficient.so", shell=True)
    check_output("rm -f radiotool/algorithms/par_build_table.c", shell=True)
    check_output("rm -f radiotool/algorithms/par_build_table.so", shell=True)
    print "trying to sudo rm -r build"
    check_output("sudo rm -r build", shell=True)
except:
    pass


