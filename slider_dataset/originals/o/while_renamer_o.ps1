while ($true) {
	$num = 1
	sleep 1
	$next = ".\o ($($num)).jpg"
	while((Test-Path -Path $next))
    		{
		    $num+=1   
		    $next = ".\o ($num).jpg"
		 }
	      mv .\o.jpg $next
        }	

   