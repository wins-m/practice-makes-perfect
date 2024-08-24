def main():
	"""Cartesian product of dictionary of lists"""
	dict0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}
	for a in dict0['a']:
		for b in dict0['b']:
			for c in dict0['c']:
				print({'a': a, 'b': b, 'c': c})


if __name__ == '__main__':
	main()
