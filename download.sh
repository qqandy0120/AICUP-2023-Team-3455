# download merge train and test data
gdown --folder 1ar3poxygTdnm2Q7uJp0WEp0ydBw7fAak
v aicup_final_data/* ./
rm -r aicup_final_data/
unzip data/all.zip
mv all/ data/
rm -f data/all.zip

# download official wiki pages
gdown --folder 1kop6Pkva0oORDU9BFpmN_UOiXCLCAe0c
mv wiki-pages/ data/

# download chpt
gdown --folder 1h0K5BFSssQAv6NtoyX43GR0Dj36QylB_
mv aicup_ckpt/* ./
rm -r aicup_ckpt/
