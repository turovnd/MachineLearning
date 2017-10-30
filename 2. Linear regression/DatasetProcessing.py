
"""
"""


class DatasetProcessing(object):
    """Initialization variables"""
    def __init__(self, area, rooms, price):
        self.area = area
        self.rooms = rooms
        self.price = price

    """Метод обработки входного датасета.

    file 'dataset.txt': входной датасет, содержащий информацию о точках в виде (area,rooms,price).
    
    Args:
        filename: имя входного датасета.
            
    Returns:
        data: лист, содержащий входной датасет в виде (area,rooms,price).
    """
    @staticmethod
    def getDataset(filename):
        data = []
        file = open(filename)
        for line in file:
            area, rooms, price = line.split(',')
            data.append([int(area), int(rooms), int(price)])
        file.close()
        return data

    """ Метод разделения листа вида (area,rooms,price) на составляющие.
    
    Args:
        data: лист, содержащий входной датасет в виде (area,rooms,price).
            
    Returns:
        area: лист, содержащий area составляющую датасета.
        rooms: лист, содержащий rooms составляющую датасета.
        price: лист, содержащий price составляющую датасета.
    """
    @staticmethod
    def getSeparetedData(data):
        area = []
        rooms = []
        price = []
        for i in range(len(data)):
            area.append(data[i][0])
            rooms.append(data[i][1])
            price.append(data[i][2])
        return area, rooms, price
