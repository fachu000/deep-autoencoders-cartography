
import numpy as np


def dbm_to_natural(x_variable_dbm):
    return db_to_natural(dbm_to_db(x_variable_dbm))


def natural_to_dbm(x_variable_nat):
    return db_to_dbm(natural_to_db(x_variable_nat))


def db_to_natural(x_variable_db):
    """
        Arguments:
              x_variable_db: must be an np array
        """
    return 10 ** (x_variable_db / 10)


def natural_to_db(x_variable_nat):
    """
      Arguments:
            x_variable_nat: must be an np array
      """
    return 10 * np.log10(x_variable_nat)


def db_to_dbm(x_variable_db):
    """
      Arguments:
            x_variable_db: must be an np array
      """
    return x_variable_db + 30


def dbm_to_db(x_variable_dbm):
    """
      Arguments:
            x_variable_dbm: must be an np array
      """
    return x_variable_dbm - 30



                
