while ($true) {
	$num = 1
	sleep 1
	$next = ".\i ($($num)).jpg"
	while((Test-Path -Path $next))
    		{
		    $num+=1   
		    $next = ".\i ($num).jpg"
		 }
	      mv .\i.jpg $next
        }	

   