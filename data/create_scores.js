use school
db.students.drop();
types = ['exam', 'quiz', 'homework', 'homework'];
for (i = 0; i < 10000000; i++) {
    scores = []
    for (j = 0; j < 4; j++) {
	scores.push({'type':types[j],'score':Math.random()*100});
    }
    record = {'student_id':i, 'scores':scores};
    db.students.insert(record);

}
	    
