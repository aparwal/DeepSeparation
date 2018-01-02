# from https://github.com/sigsep/sigsep-mus-io/blob/master/scripts/decode.sh
# provided by sigsep
# MP4Box -version
# cd "/data"

subsets=( "test" "train" )

for t in "${subsets[@]}"
do
  # encode to AAC
  cd $t;
  for stem in *.stem.mp4;
    do name=`echo $stem | cut -d'.' -f1`;
    echo "$stem";
    mkdir "$name";
    cd "$name";
    ffmpeg -loglevel panic -i "../${stem}" -map 0:0 -vn mixture.wav
    ffmpeg -loglevel panic -i "../${stem}" -map 0:1 -vn drums.wav
    ffmpeg -loglevel panic -i "../${stem}" -map 0:2 -vn bass.wav
    ffmpeg -loglevel panic -i "../${stem}" -map 0:3 -vn other.wav
    ffmpeg -loglevel panic -i "../${stem}" -map 0:4 -vn vocals.wav
    cd ..;
    rm "$stem"
  done
  cd ..;
done