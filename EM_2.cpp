#include <iostream>

int main()
{
	int x1, x2, x3, x4;

	float pik, x2k;
	std::cout << "enter vector of values, there should be four inputs\n";
	std::cin >> x1 >> x2 >> x3 >> x4;

	std::cout<<"enter value for pik\n";
	std::cin>>pik;

	for (int counter = 0; counter < 10; counter++)
	{
		x2k=x1*((0.25)*pik)/((0.5)+(0.25)*pik);
		pik=(x2k + x4)/(x2k + x2 + x3 + x4);
		std::cout << "x2k is" << x2k << "and" << "pik is" << pik << std::endl;
	}
return 0;
}
