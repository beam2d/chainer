import collections
import contextlib
import copy
import warnings

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import initializers


def _is_shape(value):
    if value is None:
        return True
    elif isinstance(value, collections.Sequence):
        try:
            return all(int(x) for x in value)
        except TypeError:
            return False
    try:
        return int(value)
    except TypeError:
        return False


def _ensure_shape_dtype(value):
    # Return value paired with dtype FP32 if it is a shape.
    if _is_shape(value):
        return value, 'f'
    # Otherwise, returns it with assuming a shape-dtype pair.
    else:
        return value


def _warn_add_param():
    warnings.warn('''\
Parameter registeration via Link.__init__ and Link.add_param is deprecated.
Assign a Parameter object directly to an attribute within a \
"with link.init_scope():" block instead.
''', DeprecationWarning)


class Link(object):

    """Building block of model definitions.

    Link is a building block of neural network models that support various
    features like handling parameters, defining network fragments,
    serialization, composition, etc.

    Link is the primitive structure for the model definitions. It supports
    management of parameter variables, *child links*, and *persistent values*.

    *Parameter* is an instance of :class:`~chainer.Parameter` registered to a
    link. A :class:`~chainer.Parameter` object can be registered as a
    parameter of the link by assigning it to an attribute within *an
    initialization scope*, which is a code surrounded by a
    :meth:`init_scope` context manager using the ``with`` statement.

    *Child links* are other :class:`~chainer.Link` objects that belong to the
    link (this link is also called the *parent link* of these child links).
    They can be registered in the same way as parameters.

    .. note::
       Before v4, :class:`~chainer.Chain` class was used to compose links. This
       class is still available for backward compatibility, though there is no
       distinction between :class:`~chainer.Link` and :class:`~chainer.Chain`
       anymore.

    *Persistent values* are arrays, scalars, or any other serializable values
    registered via :meth:`register_persistent` or :meth:`add_persistent`.

    .. note::
       Whereas arbitrary serializable objects can be registered as persistent
       values, it is strongly recommended to just register values that should
       be treated as results of learning. A typical example of persistent
       values is ones computed during training and required for testing, e.g.
       running statistics for batch normalization.

    Parameters, child links, and persistent values are referred by their names.
    They can be accessed as attributes of the link. Link class itself manages
    the lists of their names to distinguish parameters, child links, and
    persistent values from other attributes.

    Child links provide object-style composition. Another way to compose
    several links is to use :class:`ChainList`, which provides list-like
    interface to combine multiple links.

    As noted above, Link supports the serialization protocol of the
    :class:`~chainer.Serializer` class. **Note that only parameters, persistent
    values, and those of the child links are saved and loaded.** Other
    attributes are considered as a part of user program (i.e. a part of network
    definition). In order to construct a link from saved file, it is user's
    responsibility to correctly reconstruct other attributes.

    .. admonition:: Example

       This is a simple example of custom link definition. Chainer itself also
       provides many links defined under the :mod:`~chainer.links` module. They
       might serve as examples, too.

       Consider we want to define a simple primitive link that implements a
       fully-connected layer based on the :func:`~functions.linear` function.
       Note that this function takes input units, a weight variable, and a bias
       variable as arguments. Then, the fully-connected layer can be defined as
       follows::

          import chainer
          import chainer.functions as F
          from chainer import initializers
          import numpy as np

          class LinearLayer(chainer.Link):

              def __init__(self, n_in, n_out):
                  super(LinearLayer, self).__init__()
                  with self.init_scope():
                      self.W = chainer.Parameter(
                          initializers.Normal(), (n_out, n_in))
                      self.b = chainer.Parameter(
                          initializers.Zero(), (n_out,))

              def __call__(self, x):
                  return F.linear(x, self.W, self.b)

       This example shows that a user can define arbitrary parameters and use
       them in any methods. Links typically implement the ``__call__``
       operator, although they can also provide other methods to implement the
       forward propagation.

    .. admonition:: Example

       This is another example. In this example, we write a link that consists
       of other links.

       Consider we want to define a multi-layer perceptron consisting of two
       hidden layers with rectifiers as activation functions. We can use the
       the above ``LinearLayer`` as a building block::

          class MultiLayerPerceptron(chainer.Link):

              def __init__(self, n_in, n_hidden, n_out):
                  super(MultilayerPerceptron, self).__init__()
                  with self.init_scope():
                      self.layer1 = LinearLayer(n_in, n_hidden)
                      self.layer2 = LinearLayer(n_hidden, n_hidden)
                      self.layer3 = LinearLayer(n_hidden, n_out)

              def __call__(self, x):
                  # Forward propagation
                  h1 = F.relu(self.layer1(x))
                  h2 = F.relu(self.layer2(h1))
                  return self.layer3(h2)

       Child links are registered via the assignment within a
       ``with self.init_scope():`` block. The forward propagation is often
       implemented as the ``__call__`` operator as the above example, though
       it is not mandatory.

    Note that the ``LinearLayer`` defined in the above example is just a
    demonstration. It is recommended to use :class:`chainer.links.Linear`
    instead of defining new one in most cases.

    Args:
        objs: *(deprecated since v2.0.0)* Parameters or child links to register
            to the link. Parameter can also be specified by its shape (and
            optionally its dtype), with which a :class:`~chainer.Parameter`
            object is automatically generated and registered. To specify the
            dtype of the parameter, pass a tuple of the shape and dtype.

    Attributes:
        ~Link.name (str): Name of this link, given by the parent chain (if
            exists).

    """

    def __init__(self, **params):
        self._children = set()
        self._params = set()
        self._persistent = set()
        self._cpu = True
        self._device_id = None
        self._within_init_scope = False
        self.name = None

        for name, value in six.iteritems(params):
            # NOTE: deprecation warning will be raised in each method
            if isinstance(value, Link):
                self.add_link(name, link)
            elif isinstance(value, chainer.Parameter):
                _warn_add_param()
                self._add_param(name, param)
            else:
                shape, dtype = _ensure_shape_dtype(value)
                self.add_param(name, shape, dtype=dtype)

    @property
    def xp(self):
        """Array module for this link.

        Depending on which of CPU/GPU this link is on, this property returns
        :mod:`numpy` or :mod:`cupy`.

        """
        return numpy if self._cpu else cuda.cupy

    @property
    def within_init_scope(self):
        """True if the current code is inside of an initialization scope.

        See :meth:`init_scope` for the details of the initialization scope.

        """
        return getattr(self, '_within_init_scope', False)

    @contextlib.contextmanager
    def init_scope(self):
        """Creates an initialization scope.

        This method returns a context manager object that enables registration
        of parameters (and links for :class:`~chainer.Chain`) by an assignment.
        A :class:`~chainer.Parameter` object can be automatically registered
        by assigning it to an attribute under this context manager.

        .. admonition:: Example

           In most cases, the parameter registration is done in the
           initializer method. Using the ``init_scope`` method, we can
           simply assign a :class:`~chainer.Parameter` object to register
           it to the link.

           .. code-block:: python

              class MyLink(chainer.Link):
                  def __init__(self):
                      super().__init__()
                      with self.init_scope():
                          self.W = chainer.Parameter(0, (10, 5))
                          self.b = chainer.Parameter(0, (5,))

        """
        old_flag = self.within_init_scope
        self._within_init_scope = True
        try:
            yield
        finally:
            self._within_init_scope = old_flag

    def __getitem__(self, name):
        """Equivalent to getattr."""
        return getattr(self, name)

    def __setattr__(self, name, value):
        if self.within_init_scope:
            if isinstance(value, chainer.Parameter):
                self._add_param(name, value)
            elif isinstance(value, Link):
                if hasattr(self, name):
                    raise AttributeError(
                        'cannot register a new link %s: attribute exists'
                        % name)
                value.name = name
                self._children.add(name)
        super(Link, self).__setattr__(name, value)

    def __delattr__(self, name):
        self._children.discard(name)
        self._params.discard(name)
        self._persistent.discard(name)
        super(Link, self).__delattr__(name)

    def add_link(self, name, link):
        """Registers a child link to this chain.

        .. deprecated:: v2.0.0

           Assign the child link directly to an attribute within
           :meth:`~chainer.Chain.init_scope` instead.
           For example, the following code

           .. code-block:: python

               chain.add_link('l1', L.Linear(3, 5))

           can be replaced by the following line.

           .. code-block:: python

               with chain.init_scope():
                   chain.l1 = L.Linear(3, 5)

           The latter is easier for IDEs to keep track of the attribute's
           type.

        Args:
            name (str): Name of the child link. This name is also used as the
                attribute name.
            link (Link): The link object to be registered.

        """
        warnings.warn('''\
Child link registeration via Link.__init__ and Link.add_link is deprecated.
Assign a Link object directly to an attribute within a \
"with link.init_scope():" block instead.
        ''', DeprecationWarning)
        if name in self.__dict__:
            raise AttributeError(
                'cannot register a new link %s: attribute exists' % name)
        if not isinstance(link, Link):
            raise TypeError('cannot register a non-link object as a child')
        with self.init_scope():
            setattr(self, name, link)

    def add_param(self, name, shape=None, dtype=numpy.float32,
                  initializer=None):
        """Registers a parameter to the link.

        .. deprecated:: v2.0.0

           Assign a :class:`~chainer.Parameter` object directly to an
           attribute within :meth:`~chainer.Link.init_scope` instead.
           For example, the following code

           .. code-block:: python

               link.add_param('W', shape=(5, 3))

           can be replaced by the following assignment.

           .. code-block:: python

               with link.init_scope():
                   link.W = chainer.Parameter(None, (5, 3))

           The latter is easier for IDEs to keep track of the attribute's
           type.

        Args:
            name (str): Name of the parameter. This name is also used as the
                attribute name.
            shape (int or tuple of ints): Shape of the parameter array. If it
                is omitted, the parameter variable is left uninitialized.
            dtype: Data type of the parameter array.
            initializer: If it is not ``None``, the data is initialized with
                the given initializer. If it is an array, the data is directly
                initialized by it. If it is callable, it is used as a weight
                initializer. Note that in these cases, ``dtype`` argument is
                ignored.

        """
        _warn_add_param()
        if name in self.__dict__:
            raise AttributeError(
                'cannot register a new parameter %s: attribute exists'
                % name)
        if initializer is None:
            initializer = initializers.NaN(dtype)
        param = chainer.Parameter(initializer, shape)
        with self.init_scope():
            setattr(self, name, param)

    def add_persistent(self, name, value):
        """Registers a persistent value to the link.

        The registered value is saved and loaded on serialization and
        deserialization. The value is set to an attribute of the link.

        Args:
            name (str): Name of the persistent value. This name is also used
                for the attribute name.
            value: Value to be registered.

        """
        d = self.__dict__
        if name in d:
            raise AttributeError(
                'cannot register a new persistent value %s: attribute exists'
                % name)
        self._persistent.add(name)
        self._params.discard(name)
        d[name] = value

    def register_persistent(self, name):
        """Registers an attribute of a given name as a persistent value.

        This is a convenient method to register an existing attribute as a
        persistent value. If ``name`` has been already registered as a
        parameter, this method removes it from the list of parameter names
        and re-registers it as a persistent value.

        Args:
            name (str): Name of the attribute to be registered.

        """
        if not hasattr(self, name):
            raise AttributeError(
                'cannot register non-existent attribute %s as a persistent '
                'value' % name)
        self._persistent.add(name)
        self._params.discard(name)

    def copy(self):
        """Copies the link hierarchy to new one.

        The whole hierarchy rooted by this link is copied. The copy is
        basically shallow, except that the parameter variables are also
        shallowly copied. It means that the parameter variables of copied one
        are different from ones of original link, while they share the data and
        gradient arrays.

        The name of the link is reset on the copy, since the copied instance
        does not belong to the original parent chain (even if exists).

        Returns:
            Link: Copied link object.

        """
        ret = copy.copy(self)
        ret._children = set(self._children)
        ret._params = set(self._params)
        ret._persistent = set(self._persistent)
        ret.name = None
        d = ret.__dict__
        for name in ret._children:
            copied = d[name].copy()  # copy child links recursively
            copied.name = name
            d[name] = copied
        for name in ret._params:
            copied = copy.copy(d[name])
            copied.grad = None
            d[name] = copied
        return ret

    def to_cpu(self):
        """Copies parameter variables and persistent values to CPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to CPU, the link implementation must
        override this method to do so.

        Returns: self

        """
        if self._cpu:
            return self
        d = self.__dict__
        for name in self._children:
            d[name].to_cpu()
        for name in self._params:
            d[name].to_cpu()
        for name in self._persistent:
            value = d[name]
            if isinstance(value, cuda.ndarray):
                d[name] = value.get()
        self._cpu = True
        self._device_id = None
        return self

    def to_gpu(self, device=None):
        """Copies parameter variables and persistent values to GPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to GPU, the link implementation must
        override this method to do so.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        Returns: self

        """
        cuda.check_cuda_available()
        if not self._cpu:
            return self
        d = self.__dict__
        with cuda._get_device(device):
            for name in self._children:
                d[name].to_gpu()
            for name in self._params:
                d[name].to_gpu()
            for name in self._persistent:
                value = d[name]
                if isinstance(value, numpy.ndarray):
                    d[name] = cuda.to_gpu(value)
            self._device_id = cuda.cupy.cuda.get_device_id()
        self._cpu = False
        return self

    def params(self, include_uninit=True):
        """Returns a generator of all parameters under the link hierarchy.

        Args:
            include_uninit (bool): If ``True``, it also generates uninitialized
                parameters.

        Returns:
            A generator object that generates all parameters.

        """
        d = self.__dict__
        for name in self._params:
            if include_uninit or d[name].data is not None:
                yield d[name]
        for name in self._children:
            for param in d[name].params(include_uninit):
                yield param

    def namedparams(self, include_uninit=True):
        """Returns a generator of all (path, param) pairs under the hierarchy.

        Args:
            include_uninit (bool): If ``True``, it also generates uninitialized
                parameters.

        Returns:
            A generator object that generates all (path, parameter) pairs. The
            paths are relative from this link.

        """
        d = self.__dict__
        for name in self._params:
            if include_uninit or d[name].data is not None:
                yield '/' + name, d[name]
        for name in self._children:
            prefix = '/' + name
            for path, param in d[name].namedparams(include_uninit):
                yield prefix + path, param

    def links(self, skipself=False):
        """Returns a generator of all links under the hierarchy.

        Args:
            skipself (bool): If ``True``, then the generator skips this link
                and starts with the first child link.

        Returns:
            A generator object that generates all links.

        """
        if not skipself:
            yield self
        d = self.__dict__
        for name in self._children:
            for link in d[name].links():
                yield link

    def namedlinks(self, skipself=False):
        """Returns a generator of all (path, link) pairs under the hierarchy.

        Args:
            skipself (bool): If ``True``, then the generator skips this link
                and starts with the first child link.

        Returns:
            A generator object that generates all (path, link) pairs.

        """
        if not skipself:
            yield '/', self
        d = self.__dict__
        for name in self._children:
            child = d[name]
            prefix = '/' + name
            yield prefix, child
            for path, link in d[name].namedlinks(True):
                yield prefix + path, link

    def children(self):
        """Returns a generator of all child links.

        Returns:
            A generator object that generates all child links.

        """
        d = self.__dict__
        for name in self._children:
            yield d[name]

    def copyparams(self, link):
        """Copies all parameters from given link.

        This method copies data arrays of all parameters in the hierarchy. The
        copy is even done across the host and devices. Note that this method
        does not copy the gradient arrays.

        Args:
            link (Link): Source link object.

        """
        src = link.__dict__
        dst = self.__dict__
        for name in self._params:
            dst[name].copydata(src[name])
        for name in self._children:
            dst[name].copyparams(src[name])

    def cleargrads(self):
        """Clears all gradient arrays.

        This method should be called before the backward computation at every
        iteration of the optimization.

        """
        for param in self.params():
            param.cleargrad()

    def zerograds(self):
        """Initializes all gradient arrays by zero.

        This method can be used for the same purpose of cleargrads, but less
        efficient. This method is left for backward compatibility.

        .. deprecated:: v1.15
           Use :meth:`cleargrads` instead.

        """
        warnings.warn(
            'Link.zerograds is deprecated. Use Link.cleargrads instead.',
            DeprecationWarning)
        for param in self.params():
            param.zerograd()

    def addgrads(self, link):
        """Accumulates gradient values from given link.

        This method adds each gradient array of the given link to corresponding
        gradient array of this link. The accumulation is even done across
        host and different devices.

        Args:
            link (Link): Source link object.

        """
        src = link.__dict__
        dst = self.__dict__
        for name in self._params:
            dst[name].addgrad(src[name])
        for name in self._children:
            dst[name].addgrads(src[name])

    def enable_update(self):
        """Enables update rules of all parameters under the link hierarchy.

        This method sets the :attr:`~chainer.UpdateRule.enabled` flag of the
        update rule of each parameter variable to ``True``.

        """
        for param in self.params():
            rule = param.update_rule
            if rule is not None:
                rule.enabled = True

    def disable_update(self):
        """Disables update rules of all parameters under the link hierarchy.

        This method sets the :attr:`~chainer.UpdateRule.enabled` flag of the
        update rule of each parameter variable to ``False``.

        """
        for param in self.params():
            rule = param.update_rule
            if rule is not None:
                rule.enabled = False

    @property
    def update_enabled(self):
        """``True`` if at least one parameter has an update rule enabled."""
        for param in self.params():
            rule = param.update_rule
            if rule is not None and rule.enabled:
                return True
        return False

    def serialize(self, serializer):
        """Serializes the link object.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        d = self.__dict__
        for name in self._params:
            param = d[name]
            data = serializer(name, param.data)
            if param.data is None and data is not None:
                # Initialize the parameter here
                param.initialize(data.shape)
                if isinstance(param.data, numpy.ndarray):
                    numpy.copyto(param.data, data)
                else:
                    param.data.set(numpy.asarray(data))
        for name in self._persistent:
            d[name] = serializer(name, d[name])
        for name in self._children:
            d[name].serialize(serializer[name])

    def _add_param(self, name, param):
        param.name = name
        if not self._cpu:
            param.to_gpu(self._device_id)
        self._params.add(name)
        self._persistent.discard(name)


class Chain(Link):

    """Equivalent to Link.

    As of v4, :class:`~chainer.Link` class has the same functionality as Chain.
    The Chain class is left for backward compatibility.

    """
    pass


class ChainList(Link):

    """Composable link with list-like interface.

    This is a link that can be used like a list of child links. Each child link
    is indexed by a non-negative integer, and it maintains the current number
    of registered child links. The :meth:`add_link` method inserts a new link
    at the end of the list. It is useful to write a chain with arbitrary number
    of child links, e.g. an arbitrarily deep multi-layer perceptron.

    Note that this class does not implement all methods of :class:`list`.

    Args:
        links: Initial child links.

    """

    def __init__(self, *links):
        super(ChainList, self).__init__()
        self._list_children = []

        for link in links:
            self.append(link)

    def __setattr__(self, name, value):
        if self.within_init_scope and isinstance(value, Link):
            raise TypeError(
                'cannot register a new link'
                ' within a "with chainlist.init_scope():" block.')
        super(ChainList, self).__setattr__(name, value)

    def __getitem__(self, index):
        """Returns the child at given index.

        Args:
            index (int): Index of the child in the list.

        Returns:
            Link: The ``index``-th child link.

        """
        return self._list_children[index]

    def __iter__(self):
        return iter(self._list_children)

    def __len__(self):
        """Returns the number of children."""
        return len(self._list_children)

    def append(self, link):
        """Registers a child link and adds it to the tail of the list.

        This is equivalent to :meth:`add_link`. This method has been added to
        emulate the ``list`` interface.

        Args:
            link (Link): The link object to be regsitered.

        """
        self.add_link(link)

    def add_link(self, link):
        """Registers a child link and adds it to the tail of the list.

        Args:
            link (Link): The link object to be registered.

        """
        link.name = str(len(self._list_children))
        self._list_children.append(link)

    def copy(self):
        ret = super(ChainList, self).copy()
        ret._list_children = list(ret._list_children)  # copy
        children = ret._list_children
        for i, child in enumerate(children):
            child = child.copy()
            child.name = str(i)
            children[i] = child
        return ret

    def to_cpu(self):
        super(ChainList, self).to_cpu()
        for link in self._list_children:
            link.to_cpu()
        return self

    def to_gpu(self, device=None):
        with cuda._get_device(device):
            super(ChainList, self).to_gpu()
            for link in self._list_children:
                link.to_gpu()
        return self

    def params(self, include_uninit=True):
        for param in super(ChainList, self).params(include_uninit):
            yield param
        for link in self._list_children:
            for param in link.params(include_uninit):
                yield param

    def namedparams(self, include_uninit=True):
        for ret in super(ChainList, self).namedparams(include_uninit):
            yield ret
        for idx, link in enumerate(self._list_children):
            prefix = '/%d' % idx
            for path, param in link.namedparams(include_uninit):
                yield prefix + path, param

    def links(self, skipself=False):
        if not skipself:
            yield self
        for child in self._list_children:
            for link in child.links():
                yield link

    def namedlinks(self, skipself=False):
        if not skipself:
            yield '/', self
        for idx, child in enumerate(self._list_children):
            prefix = '/%d' % idx
            yield prefix, child
            for path, link in child.namedlinks(True):
                yield prefix + path, link

    def children(self):
        for child in self._list_children:
            yield child

    def copyparams(self, link):
        super(ChainList, self).copyparams(link)
        for idx, child in enumerate(self._list_children):
            child.copyparams(link[idx])

    def addgrads(self, link):
        super(ChainList, self).addgrads(link)
        for idx, child in enumerate(self._list_children):
            child.addgrads(link[idx])

    def serialize(self, serializer):
        super(ChainList, self).serialize(serializer)
        for idx, child in enumerate(self._list_children):
            child.serialize(serializer['%d' % idx])
