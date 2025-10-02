# Deep Learning Approach to BoR

## Segmentation of Surface
- This may or may not be nessecary, but to simplify this I can buy a practice pad for us to test this on.
- We could segment the pad using edge detection and fit an elipse to it, similar to how it was done in the paper

## Tracking Stick Tips: 
- This is an interesting problem because we don't have a High FPS camera available, so our images will have some blur. Will need to collect good data and some averaging.
- We can do manual differentiation between the sticks by coloring the tips, red and blue. This solves persistance issues.

## Hit Detection:
- Information about the acceleration, direction, and velocity can be added to an RNN/LSTM, something good with time sequence.
- Another option is to use Optical Flow to do this.  This is a simple and quick way of tracking tip velocity and motion vectors. 

# Next Steps:
- Improving Stick Detection and Segmentation
- Hit Detection
- Implementing a Model for Transcription.
- Is this possible to track real time?