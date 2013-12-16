
use test
for (var j = 0; j < 10000; j++){
    db.foo.insert({'a':j,'b':j, 'c':j});
}

