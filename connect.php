<?php
$email=$_POST['email'];
$pwd=$_POST['pwd'];


//Database connection
$conn=new mysqli('localhost','root','','test');
if($conn->connect_error){
	die('connection Failed :' .$conn->connect_error)
	else
	{
		$stmt=$conn->prepare("insert into registration(email,pwd)value(?,?)");
		$stmt->bind_param("ss",$email,$pwd)
		$stmt->execute();
		
		$stmt->close();
		$conn->close();
	}
}

?>