## Getting Started with HW3B

### Step 1: Update Your Class Repository

The assignment files are now available in the class repository. Open your terminal and update your local copy:

```bash
cd ~/insy6500/class_repo
git pull
```

You should see output indicating new files were downloaded. If it says "Already up to date", the files may have been added previously.

### Step 2: Copy Assignment Files to Your Work Repository

You'll work on this assignment in your own repository (`my_repo`), not in the class repository. Copy the necessary files with your system file explorer or with the following terminal commands:

```bash
# Copy the notebook
cp ~/insy6500/class_repo/homework/hw3b.ipynb ~/insy6500/my_repo/homework/

# Copy the data file
cp ~/insy6500/class_repo/data/202410-divvy-tripdata-100k.csv ~/insy6500/my_repo/data/
```

Note that the data file should be in a data subdirectory. You may need to create that first.

Verify the files copied successfully:

```bash
cd ~/insy6500/my_repo
ls homework/hw3b.ipynb
ls 202410-divvy-tripdata-100k.csv
```

Both commands should show the file exists.

### Step 3: Open the Notebook

Start Jupyter Lab from your project directory with your `insy6500` environment activated:

```bash
cd ~/insy6500
conda activate insy6500
jupyter lab
```

In Jupyter Lab's file browser, navigate to `my_repo/homework/` and open `hw3b.ipynb`.

### Step 4: Work on the Assignment

Complete the assignment in your copy of the notebook. Remember:

- **Work in `my_repo`**, not `class_repo` - the class repo is read-only reference material
- The notebook expects the data file at `../data/202410-divvy-tripdata-100k.csv` (relative to the homework folder)
- Save your work frequently (`Cmd+S` or `Ctrl+S`)

### Step 5: Commit and Push Your Work

As you make progress, commit your changes to your GitHub repository:

```bash
cd ~/insy6500/my_repo
git add homework/hw3b.ipynb
git commit -m "Working on hw3b - completed Problem 1"
git push
```

This will commit changes to `my_repo` and push them to your GitHub account.

You can commit multiple times as you work through the assignment. Before your final submission, ensure:

- Your notebook runs completely with "Kernel → Restart & Run All"
- All interpretation questions are answered
- Your final version is committed and pushed

### Submit Your Assignment

After pushing your finished work to GitHub (step 5), find the HW3b assignment on Canvas and submit the URL to your repo. We will grab your submission from GitHub for grading.

### Troubleshooting

**Problem**: `FileNotFoundError` when loading the CSV  
**Solution**: Check your working directory. The notebook should be in `my_repo/homework/` and the data in `my_repo/data/`. The path `'../data/202410-divvy-tripdata-100k.csv'` goes up one level from homework to find data.

**Problem**: Changes to `class_repo` files  
**Solution**: Don't modify files in `class_repo`. If you accidentally did, run `cd ~/insy6500/class_repo && git restore .` to reset them, then copy fresh versions to `my_repo`.

**Problem**: Lost track of which repo you're in  
**Solution**: Use `pwd` to see your current directory. You should work in `~/insy6500/my_repo`, not `~/insy6500/class_repo`.

