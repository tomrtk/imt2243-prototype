"""
    World view class(abstract) and sub classes for a position model
    and external environment model.

    Each subclass defines a coordinate system to populate with its
    corresponding objects.

"""

class WorldView:
    pass


class PositionModel(WorldView):
    pass


class EnviromentModel(WorldView):

    """EnviromentModel for all external object tracked

    Coordinate system for all external object tracked.
    Objects stored in list.

    """

    def __init__(self, refPoint, datum):
        self.__referencePoint = refPoint # Internal origo reference
                                         # point to real world coordinates
        self.__referencePointDatum = datum # Datum of ref. Point

        __objectList = []
        __objectsInModel = 0


    def addObject(object):

        """Adds a new object to __objektlist

        Add the object sent as parameter to the beginning of the list

        """

        __objectList.insert(0, object)
        __objectsInModel += 1




class ObjectDetection:

    def detectObjecti(data):
        pass


class VisualDetection(ObjectDetection):
    def detectObject(frame):
        pass



class Object:
    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z




