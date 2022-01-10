# https://www.kaggle.com/wolfram77/puzzlef-pagerank-cuda
import os
from IPython.display import FileLink
src="pagerank-cuda"
inp="/kaggle/input/graphs"
out="{}.txt".format(src)
!printf "" > "$out"
display(FileLink(out))
!ulimit -s unlimited && echo ""
!nvidia-smi && echo ""

# Download program
!rm -rf $src
!git clone https://github.com/puzzlef/$src
!echo ""

# Run
!nvcc -std=c++17 -Xcompiler -DNVGRAPH_DISABLE -O3 $src/main.cu
!stdbuf --output=L ./a.out $inp/web-Stanford.mtx      2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/web-BerkStan.mtx      2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/web-Google.mtx        2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/web-NotreDame.mtx     2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/soc-Slashdot0811.mtx  2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/soc-Slashdot0902.mtx  2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/soc-Epinions1.mtx     2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/coAuthorsDBLP.mtx     2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/coAuthorsCiteseer.mtx 2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/soc-LiveJournal1.mtx  2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/coPapersCiteseer.mtx  2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/coPapersDBLP.mtx      2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/indochina-2004.mtx    2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/italy_osm.mtx         2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/great-britain_osm.mtx 2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/germany_osm.mtx       2>&1 | tee -a "$out"
!stdbuf --output=L ./a.out $inp/asia_osm.mtx          2>&1 | tee -a "$out"
