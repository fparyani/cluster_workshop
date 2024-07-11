# Cluster Workshop

## Activity assignment

1. ssh into Manitou via the following command `ssh user@128.59.124.102` or `ssh user@128.59.124.103`

2. Explore file directory through basic bash commands

3. Create a directory in `/burg/pmg/users/` with your uni as the directory name

4. Create a conda environment via 

```bash
module load mamba

mamba create --prefix /*in the directory in 3*/*name of env*
```

5. Install jupyter through `pip install jupyter` or `pip install jupyter-lab` command and launch notebook

6. Run example scripts in this doc

7. Save output, copy the file back to your home directory, and transfer it to your local computer through the `scp` command 

Extra exercise

1. Try running a batch script via `sbatch` command on my sample cellpose script or try running your own script!

2. Download monitoring scripts off Manitou github repo and run it on your home directory to learn more about Manitou live activity
