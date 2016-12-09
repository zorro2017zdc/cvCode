function tmp=which_min3(x,y,z)
if(min([x,y,z])==y)
    tmp=1;
elseif(min([x,y,z])==x)
    tmp=0;
    else
    tmp=2;
end