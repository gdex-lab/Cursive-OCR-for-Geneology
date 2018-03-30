while ($true) {
	$num = 1
	sleep 1
	$next = ".\a ($($num)).jpg"
	while((Test-Path -Path $next))
    		{
		    $num+=1   
		    $next = ".\a ($num).jpg"
		 }
	      mv .\a.jpg $next
        }	

   