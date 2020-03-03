function strimage(a)
  lena = size(a);
  lena = lena(2);
  xy = sscanf(a(4:lena), '%d:%d');
  lenxy = size(xy);
  lenxy = lenxy(1);
  grid = [];
  grid(784) = 0;
  for i=2:2:lenxy
    grid(xy(i-1)) = xy(i) * 100/255;
  end
  image(reshape(grid,28,28))
end

