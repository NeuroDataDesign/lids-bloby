# GMM Threshold Pseudocode

## Old Algorithm

1. Read image into an array
2. Take the unique intensity values and find out the counts of each unique intensity and call it `data_points`. So

```
data_points = [(I1, C1), (I2, C2), (I3, C3) ... (In, Cn)]
```

where I is intensity and C is count.

3. Fit `data_points` using a 2 or 3 component GMM (based on whether the data source is `laVision` or `COLM`) and take the
each cluster's mean intensity.

4. Choose the `threshold` as the mean of the last 2 clusters' intensities.

![gmm1](https://user-images.githubusercontent.com/1017519/36445537-9b46d7b8-164c-11e8-918b-fa1af892dea6.png)

From the above image we can see that the GMM is doing a poor job. Hence we go to the new algorithm which is explained here.

## New Algorithm (Changed on 02/19/2018)

1. Read image into an array
2. Take _all_ intensity values of the image and let's call it `all_intensities` which is

```
all_intensites = [I1, I2, I3, ...... In]
```

3. Reshape the above 1D array into a 2D array because GMM needs at least a 2D array and let's call it `reshaped_intensities`
which is

```
reshaped_intensities = [[I1], [I2], [I3], .... [In]]
```

4. Fit `reshaped_intensities` into a 4 or 5 component GMM (based on whether the data source is `laVision` or `COLM`) and
take the GMM model's cluster means and call it `cluster_means`.

5. Sort `cluster_means` and take the last element as `threshold`.

**Note**: In the new method, number of components is more because GMM now has more information and the model fit is more fine
grained and the intensities are put into more fine grained bins.

![gmm2](https://user-images.githubusercontent.com/1017519/36445540-9f364b24-164c-11e8-8f18-0c79dd8b8fcd.png)
