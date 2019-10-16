import test

def main():
    pop = [test.random_point() for _ in range(test.N_POPULATION)]
    test.RPTree(pop, 0, None)


if __name__ == "__main__":
    main()
