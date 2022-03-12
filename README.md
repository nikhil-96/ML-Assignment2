## Instructions for completing the assignment
* Clone this repo to your local machine (or a hosted service such as Google Colab)
* Answer the questions in the notebook 'Assignment_2.ipynb'
* Don't edit anything in the `submit` folder
* Make sure that all answers are self-contained. Define additional variables and helper functions inside the functions you are asked to implement
* On the command line, go into the 'submit' folder and run the verification file
```
cd submit
python verify.py
```

* Check the output for any other errors. See the notebook `submit/Template.ipynb` for more details on any errors.
* A file `Submission.html` will be generated in this folder. Make sure that your answers are complete in this report.
* Push your solution Github Classroom. Make sure that you commit everything, especially your assignment notebook and submission HTML files.
```
git add -A
git commit -m "commit message"
git push
```
* An automated unit test will be run after your submission. Check your github page to follow status. A green check means that everything is fine. 


### Further tips:
* We recommend committing an empty solution right away (e.g. by changing a single character) so that you are familiar with the submission procedure. This avoids unexpected surprises near the submission deadline.
* If the import of your code fails, often signaled by the error "NameError: name 'solution' is not defined", try to see what is wrong by running
```
python solution.py
```
* If the automated test fails (you see a red cross), click it to check what went wrong. 
* If the automated test failed, that does not automatically mean that your solution is wrong. There can be other unexpected reasons as well. One reason could be that you run verify.py in a different environment, or that there is a problem with your local environment. Try pushing to github and see if the automated check passes.
* The automated test does not check for correctness, only that we could successfully run your submitted solution.
* If you needed any additional libraries, add them to requirements.txt, so that they are installed before runnign the test.
