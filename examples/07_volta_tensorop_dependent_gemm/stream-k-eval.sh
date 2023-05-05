for M in 1024 2048  
do 
	echo $M	
	./stream-k --m=$M --n=4608 --k=12288 --alpha=1 --beta=0 --split=1 --iterations=20
       	echo
	./stream-k --m=$M --n=12288 --k=1536 --alpha=1 --beta=0 --split=1 --iterations=20 
done
