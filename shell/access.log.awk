awk 'BEGIN{FPAT="[^ ]*|(\"[^\"]*\")|(\\[[^]]*\\])"}{print $0;for (i=1;i<NF;i++){print "<"$i">"}print "-----------"NF}' access.log
