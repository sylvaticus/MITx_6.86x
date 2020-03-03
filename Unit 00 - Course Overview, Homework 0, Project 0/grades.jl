# ------------------------------------------------------------------------------
# A Julia script to compute your exact grade according to the course's rules
# ------------------------------------------------------------------------------


using Statistics

#=
Grading policy

Your overall score in this class will be a weighted average of your scores for the different components, with the following weights:

16% for the lecture exercises (divided equally among the 16 out of 19 lectures)
18% for the homeworks (divided equally among 6 (out of 7) homeworks)
2% for the Project 0
36% for the Projects (divided equally among 4 (out of 5)
12% for the Midterm exam (timed)
16% for the final exam (timed)

To earn a verified certificate for this course, you will need to obtain an overall score of 60% or more of the maximum possible overall score.

Lecture Exercises, Problem Sets, and Projects

    The lowest 3 scores among the 19 lectures will be dropped, so only 16 out of 19 lectures will count .

    The lowest 1 scores among the 7 homeworks will be dropped, so only 6 out of 7 homeworks will count .

    The lowest 1 score among the 5 projects (excluding project 0) will be dropped, so only 4 out of 5 projects will count .
=#

lectures_grades  = [0/11,0/18,0/12,0/11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
homeworks_grades = [0/61,0/28,0/12,0,0,0,0]
projects_grades  = [0/39,0,0,0,0] # exclude Project 0 here
project0_grade   = 0
midterm_grade    = 0
final_grade      = 0
lectures_grades_retained=sort(lectures_grades)[4:end]
homeworks_grades_retained=sort(homeworks_grades)[2:end]
projects_grades_retained=sort(projects_grades)[2:end]

grades = [mean(lectures_grades_retained),mean(homeworks_grades_retained),project0_grade,mean(projects_grades_retained),midterm_grade,final_grade]

w_lectures  = 0.16
w_homeworks = 0.18
w_project0  = 0.02
w_projects  = 0.36
w_midterm   = 0.12
w_final     = 0.16
weigths     = [w_lectures,w_homeworks,w_project0,w_projects,w_midterm,w_final]

overallGrade = grades' * weigths
