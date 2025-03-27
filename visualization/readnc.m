function out = readnc(fname)

info = ncinfo(fname);

for i = 1:length(info.Variables)
    name = info.Variables(i).Name;
    
    eval(['out.',name,' = ncread(''',fname,''',''',name,''');']);
    
end