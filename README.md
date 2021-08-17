# Open Source Super Sampling (OSSS)

## What is this?
This is DLSS, but probs worse and trained very poorly. Able to get < 16ms/frame (Usually 10ms/frame on average) upscaling to 4k on a 3090, but your mileage may vary, does support non RTX cards as long as you have CUDA and CUDnn installed.

## Why use it?
'Cause it beats stuff like Waifu2X that takes aeons to upscale high res pics.

## What can you use it for?
Anything

## What has been tested?
Anything

Also space photos & video game footage.

Testing out upscaling video content from youtube and including it in the training data to make it more generalized.

## How long does it take for new refined versions/sub versions?
I'm a single man with a single 3090, so a while.

## Are there weights, or do we have to self-train?
Yeah, I'll post updated weights for the each model (Regular & small) in the releases section, once they are trained enough. There will also be comparison pictures for each release in extremely high res.

## What are the supported cards?
Any card that can support tensorflow/keras can support this upscaling method, you can also write conversion tools to convert to other platforms, e.g tensorflow lite.

This is fine tuned for NVIDIA 30xx series cards, for 4k60 upscaling in real time.

## Can I use this in my own software?
Sure, just give credit back to this repo so people know what you are using, I like transparency, if you are using my things, I expect you to uphold that.

Also this is licensed, GNU GPLv3 yo.

## Training
Take a 4k video (Preferebly, works best with high res content), run it through ffmpeg like so:
`ffmpeg -i video.mp4 frames/%09d.png`

Make sure the sub folder `frames` exists before running the command, once it is done though you'll have a folder (Large folder btw, make sure you got GBs to spare) with each frame of the video labelled from `000000000.png` to `999999999.png`.

Then run `python src/main.py --data path/to/frames/directory --input_size 3840x2160 --batch_size 2 --scale 4`

## Testing inference speeds
Edit `src/test.py` and load the weights into the model, then run the script with `python src/test.py --image path/to/image.png`.

It will run inference on the image a couple of times and then write it to the `--output` argument, or the default which is `upscaled.png`.

The inference will give you the time it took to upscale, along with the average time it took to upscale.

**WARNING**: Larger images take exponentionally longer times to upscale, these models are fine tuned for 4k60 upscaling in real time on NVIDIA 30xx series cards.
