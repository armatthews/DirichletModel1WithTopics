dm1wt: model3.cc
	g++ -std=c++11 -Wall -DDEBUG -D_DEBUG -g -rdynamic -I/home/austinma/git/cpyp/ model3.cc -o model3 -lboost_program_options-mt -lboost_serialization
#	g++ -std=c++11 -Wall -O -I/home/austinma/git/cpyp/ model3.cc -o model3 -lboost_program_options-mt -lboost_serialization
